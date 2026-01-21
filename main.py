import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import warnings
import os
import copy

from torch.utils.tensorboard import SummaryWriter
from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from utils.evaluate import evaluate, cal_train_metrics, suppression, test


warnings.simplefilter("ignore")

def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2
    else:
        args.eval_freq = 10

def set_environment(args, tlogger):

    print("Setting Environment...")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    train_loader, val_loader, test_loader = build_loader(args)

    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### = = = =  Model = = = =
    tlogger.print("Building Model....")
    ## 添加用来分类标签的
    coarse_classes = None

    ## For Moe
    if getattr(args, "use_ori_model", True):
        model = MODEL_GETTER[args.model_name](
            use_fpn = args.use_fpn,
            img_size = args.data_size,
            fpn_size = args.fpn_size,
            use_selection = args.use_selection,
            num_classes = args.num_classes,
            num_selects = args.num_selects,
            use_combiner = args.use_combiner,
            coarse_classes = coarse_classes,
            add_linear = getattr(args, "add_linear", False),
            feature_type = getattr(args, 'feature_type', None),
            add_loss = getattr(args, 'add_loss', False),
            only_loss = getattr(args, 'only_loss', False),
            no_mask = getattr(args, 'no_mask', False),
            use_embedding = getattr(args, 'use_embedding', False),
            pretrained_path = getattr(args, 'pretrained', None),
            use_cam = getattr(args, 'use_cam', False),
            num_bins=getattr(args, 'num_bins', 8),
            fuse_type=getattr(args, 'fuse_type', 0),
        ) # about return_nodes, we use our default setting
    else:
        print("=> Use original model.")
        exit(0)

    ## 统计可训练的参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练参数量: ({:.2f}M)".format(total_params / 1e6))

    start_epoch = 0

    model.to(args.device)
    tlogger.print()

    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """

    if train_loader is None:
        return train_loader, val_loader, test_loader, model, None, None, None, None

    ### = = = =  Optimizer = = = =
    tlogger.print("Building Optimizer....")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    ### = = = =  tensorboard = = = =
    summary_writer = SummaryWriter(log_dir=args.save_dir)

    return train_loader, val_loader, test_loader, model, optimizer, schedule, scaler, amp_context, start_epoch, summary_writer



def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader, writer):
    optimizer.zero_grad()
    total_batchs = len(train_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0

    n_left_batchs = len(train_loader) % args.update_freq

    for batch_id, batch in enumerate(train_loader):
        ids, datas, labels = batch
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        # temperature = (args.temperature - 1) * (get_lr(optimizer) / args.max_lr) + 1

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        if args.use_embedding:
                            criterion = nn.CrossEntropyLoss(reduction='none')
                            loss_c = (torch.exp(outs['text_weight']) * criterion(outs[name], labels)).mean()
                        else:
                            loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c
                        writer.add_scalar("loss/comb_loss", args.lambda_c * loss_c.item(), epoch * total_batchs + batch_id)

                elif "text_outs" in name:
                    if not getattr(args, "add_linear", False):
                        raise ValueError("add_linear not use here.")
                    if args.lambda_c != 0:
                        loss_t = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_t
                        writer.add_scalar("loss/text_loss", args.lambda_c * loss_t.item(), epoch * total_batchs + batch_id)

                elif "visual_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")
                    if args.lambda_c != 0:
                        loss_p_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += loss_p_c
                        writer.add_scalar("loss/visual_loss", loss_p_c.item(), epoch * total_batchs + batch_id)

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
                writer.add_scalar("loss/total_loss", loss.item(), epoch * total_batchs + batch_id)

            if batch_id < len(train_loader) - n_left_batchs:
                loss /= args.update_freq
            else:
                loss /= n_left_batchs

        """ = = = = calculate gradient = = = = """
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0 or (batch_id + 1) == len(train_loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update() # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1

def main(args, tlogger):
    """
    save model last.pt and best.pt
    """

    train_loader, val_loader, test_loader, model, optimizer, schedule, scaler, amp_context, start_epoch, writer = set_environment(args, tlogger)

    best_acc = 0.0
    best_eval_name = "null"


    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader, writer)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        eval_freq_schedule(args, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
        torch.save(checkpoint, args.save_dir + f"backup/last.pt")

        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                ## 待修改
                acc = accs['combiner-top-1']
                tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
                tlogger.print()

            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, args.save_dir + f"backup/best.pt")

    ## test模型
    if test_loader is not None:
        ckpt_path = args.save_dir + f"backup/best.pt"
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        test(args, model, test_loader, tlogger, best_eval_name)


if __name__ == "__main__":

    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)

    if args.n != "":
        args.exp_name = args.n

    if args.b != -1:
        setattr(args, 'num_bins', args.b)

    if args.f != -1:
        setattr(args, 'fuse_type', args.f)

    if args.m == 1:
        setattr(args, 'model_name', 'resnet50')
    elif args.m == 2:
        setattr(args, 'model_name', 'swin-t')
    elif args.m == 3:
        setattr(args, 'model_name', 'vit')
    elif args.m == 4:
        setattr(args, 'model_name', 'bioclip')

    tlogger.print("Config Loaded.")

    build_record_folder(args)
    tlogger.print()


    main(args, tlogger)
