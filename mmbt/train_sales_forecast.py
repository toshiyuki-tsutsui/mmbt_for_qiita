import argparse
from sklearn.metrics import f1_score, accuracy_score, recall_score, fbeta_score
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
from mmbt.data.helpers import get_data_loaders
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time
import pickle

# pickleを保存
def save_pickle(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


def get_args(
    target_col,
    bert_model,
    batch_sz,
    max_epochs,
    model,
    n_classes,
):
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "bert-large-uncased"],
    )
    parser.add_argument("--data_path", type=str, default="./data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument(
        "--glove_path", type=str, default="./glove_embeds/glove.840B.300d.txt"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument(
        "--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"]
    )
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument(
        "--model", type=str, default="bow", choices=["bert", "bow", "concatbow", "concatbert", "img", "mmbt"]
    )
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb")
    parser.add_argument(
        "--task_type",
        type=str,
        default="multilabel",
        choices=["multilabel", "classification"],
    )
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--batch_cnt", type=int, default=1)
    parser.add_argument("--i_epoch", type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument("--n_target_variables", type=int, default=2)
    args = parser.parse_args(
        args=[
            "--batch_sz",
            batch_sz,
            "--bert_model",
            bert_model,
            "--gradient_accumulation_steps",
            "40",
            "--savedir",
            "./savedir/",
            "--name",
            "mmbt_model_run",
            "--data_path",
            # "./datasets/",
            "./datasets/food101",# mmbtデータセット用
            # "--task",
            "--task_type",
            "classification",
            "--model",
            model,
            "--num_image_embeds",
            "3",
            "--freeze_txt",
            "5",
            "--freeze_img",
            "3",
            "--patience",
            "5",
            "--dropout",
            "0.1",
            "--lr",
            "5e-05",
            "--warmup",
            "0.1",
            "--max_epochs",
            max_epochs,
            "--seed",
            "1",
            "--hidden_sz",
            "768",
            "--batch_cnt",
            "1",
            "--n_classes",
            n_classes
        ]
    )
    return args


def get_criterion(args):

    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):

    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "max",
        patience=(args.lr_patience),
        verbose=True,
        factor=(args.lr_factor),
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts, preds_score = [], [], [], []

        for batch in data:
            loss, out, tgt, attention_probs = model_forward(
                i_epoch, model, args, criterion, batch
            )
            losses.append(loss.item())
            pred = (
                torch.nn.functional.softmax(out, dim=1)
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            preds.append(pred)
            pred_score = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()
            preds_score.append(pred_score)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            args.i_epoch = i_epoch

    metrics = {"loss": np.mean(losses)}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    preds_score = [l for sl in preds_score for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["recall"] = recall_score(tgts, preds, average="micro")
    # auc
    if args.n_classes == 2:
        pred_0 = [x[1] for x in preds_score]
        cleaned_pred_0 = [0 if str(x) == "nan" else x for x in pred_0]
        metrics["auc"] = roc_auc_score(tgts, cleaned_pred_0)
    if store_preds:
        store_preds_to_disk(tgts, preds, args)
    return metrics


def model_test(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts, preds_score,attention_probs = [], [], [], [], []
        for batch in data:
            loss, out, tgt, attention_prob = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())
            pred = (
                torch.nn.functional.softmax(out, dim=1)
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            preds.append(pred)
            pred_score = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()
            preds_score.append(pred_score)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            # attention_probs.append(attention_prob)

    metrics = {"loss": np.mean(losses)}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    preds_score = [l for sl in preds_score for l in sl]
    # attention_probs = [l for sl in attention_probs for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    c_report = classification_report(tgts, preds, output_dict=True)
    c_matrix = confusion_matrix(tgts, preds)

    return c_report, c_matrix, [losses, preds, tgts, preds_score],attention_probs


def model_production(data, model):
    with torch.no_grad():
        preds, preds_score = [], []
        for batch in data:
            out = model_forward_for_production(model, batch)
            pred = (
                torch.nn.functional.softmax(out, dim=1)
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            preds.append(pred)
            pred_score = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()
            preds_score.append(pred_score)

    preds = [l for sl in preds for l in sl]
    preds_score = [l for sl in preds_score for l in sl]

    return preds_score


def model_forward(i_epoch, model, args, criterion, batch):
    key, txt, segment, mask, img, tgt = batch
    freeze_img = i_epoch < args.freeze_img  # '--freeze_img', '3',
    freeze_txt = i_epoch < args.freeze_txt  # '--freeze_txt', '5',

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
        attention_probs = ''
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
        attention_probs = ''
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
        attention_probs = ''
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
        attention_probs = ''
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
        attention_probs = ''
    else:
        assert args.model == "mmbt"
        # エンコーダーを更新する処理
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = (
                not freeze_img
            )  # args.freeze_img=3の場合、freeze_imgはepoch<3 の時はTrueが入る。すると、not True → Falseになる。
            # print(('学習'+str(i_epoch)+'回目'),('param.requires_grad',param.requires_grad),('model.enc.img_encoder.parameters()',param))
        else:
            for param in model.enc.encoder.parameters():
                param.requires_grad = (
                    not freeze_txt
                )  # args.freeze_txt=5の場合、freeze_txtはepoch<5 の時はTrueが入る。すると、not True → Falseになる。
                # print(('学習'+str(i_epoch)+'回目'),('param.requires_grad',param.requires_grad),('model.enc.encoder.parameters()',param))
            else:
                txt, img = txt.cuda(), img.cuda()
                mask, segment = mask.cuda(), segment.cuda()
                out, attention_probs = model(txt, mask, segment, img)

            # パラメーター更新確認用
            # for n, p in model.named_parameters():
            #     if p.requires_grad==True:
            #         print("p.requires_grad==True")
            #         print('学習'+str(i_epoch)+'回目')
            #         print(n)
            #         g = p.grad     # 与えられた入力（x）によって計算された勾配の値（grad）を取得
            #         print(('勾配の値（grad）',g))
            #         # print(p)
            #         print("--------")
            #     if p.requires_grad==False:
            #         print("p.requires_grad==False")
            #         print('学習'+str(i_epoch)+'回目')
            #         print(n)
            #         g = p.grad     # 与えられた入力（x）によって計算された勾配の値（grad）を取得
            #         print(('勾配の値（grad）',g))
            #         print("--------")

    tgt = tgt.cuda()
    loss = criterion(out, tgt)

    return loss, out, tgt, attention_probs


def model_forward_for_production(model, batch):
    key, txt, segment, mask, img, tgt = batch
    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    out, attention_probs = model(txt, mask, segment, img)

    return out, attention_probs


# 学習＆モデル保存
def train(args, dataloaders_dict):
    start = time.time()
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    train_loader = dataloaders_dict["train"]
    val_loader = dataloaders_dict["val"]

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    # logger.info(model)
    #     model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    # ネットワークをGPUへ
    model.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        print("学習" + str(i_epoch) + "回目")
        train_losses = []
        model.train()
        optimizer.zero_grad()
        # for batch in tqdm(train_loader, total=len(train_loader)):
        for idx, batch in enumerate(train_loader):
            loss, _, _, attention_probs = model_forward(
                i_epoch, model, args, criterion, batch
            )
            # loss, _, _ = model_forward(
            #     i_epoch, model, args, criterion, batch
            # )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())
            loss.backward()
            global_step += 1

            # print(('global_step',global_step),('gradient_accumulation_steps',args.gradient_accumulation_steps))
            # if i_epoch in [0, 7]:
            #     sdict = model.state_dict()
            #     save_pickle(sdict,'./savedir/mmbt_model_run/'+ 'epoch_'+ str(i_epoch)+ '_state_dict.pkl',)
            #     save_pickle(attention_probs,'./savedir/mmbt_model_run/'+ 'epoch_'+ str(i_epoch)+ '_attention_probs.pkl',)
            #     save_pickle(batch,'./savedir/mmbt_model_run/'+ 'epoch_'+ str(i_epoch)+ '_batch.pkl',)
    
            # optimizer.step()を実行す前に実行する学習ステップ数。global_stepがgradient_accumulation_steps（40）に到達した場合にoptimizer.step()する
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        if args.n_classes == 2:
            logger.info("Val AUC: {:.4f}".format(metrics["auc"]))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = metrics["recall"] if len(args.labels) == 2 else metrics["acc"]
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        # if n_no_improve >= args.patience:
        #     logger.info("No improvement. Breaking out of loop.")
        #     break
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# testデータで予測＆評価
def test(args, save_model_name, dataloaders_dict):

    test_loader = dataloaders_dict["test"]
    model = get_model(args)
    load_checkpoint(
        model,
        "./savedir/"
        + "/mmbt_model_run/"
        + save_model_name
        + ".pt",
    )
    criterion = get_criterion(args)

    # logger = create_logger("%s/logfile.log" % args.savedir, args)
    # logger.info(model)
    #     model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ネットワークをGPUへ
    model.to(device)

    # 評価
    classification_report, confusion_matrix, detail_list,attention_probs = model_test(
        np.inf, test_loader, model, args, criterion, False
    )

    sdict = model.state_dict()
    # save_pickle(sdict,'./savedir/mmbt_model_run/test_state_dict.pkl')
    # save_pickle(attention_probs,'./savedir/mmbt_model_run/test_attention_probs.pkl')
    # save_pickle(dataloaders_dict,'./savedir/mmbt_model_run/test_dataloaders_dict.pkl')
    # save_pickle(detail_list,'./savedir/mmbt_model_run/detail_list.pkl')

    return classification_report, confusion_matrix, detail_list,attention_probs


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


# 本番データで予測＆評価
def production(args, model_name, dataloaders_dict):

    production_loader = dataloaders_dict["production"]
    model = get_model(args)
    load_checkpoint(
        model, "./savedir/mmbt_model_run/" + model_name + ".pt"
    )
    criterion = get_criterion(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ネットワークをGPUへ
    model.to(device)

    # 評価
    preds_score = model_production(production_loader, model)
    return preds_score
