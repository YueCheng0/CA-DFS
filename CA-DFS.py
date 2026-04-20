
import os, time, json, datetime, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")


def make_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "logs.txt")
    def log(msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def load_table(path, log):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, index_col=0)
    log(f"Loaded: {path} | shape={df.shape}")
    return df


def compute_modal_score_raw(X_df, y, cv=5, log=None):
    X = X_df.values
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    scores = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], preds))
    score = float(np.mean(scores))
    if log:
        log(f"BAcc modal score ({cv}-fold): {score:.4f}")
    return score

class GroupSparseDFS(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, per_feature_group_ids, dropout_rate):
      
        super().__init__()


        self.feature_mask = nn.Parameter(torch.ones(input_dim))
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
        self.groups = torch.tensor(per_feature_group_ids, dtype=torch.long)

    def forward(self, x):
        x = x * self.feature_mask
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x) 
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

def dfs_train_eval(
    X_train, y_train, X_test, y_test,
    per_feature_group_ids, n_classes,
    n_epochs, lr, l1_strength, group_strength,
    hidden_dim1, hidden_dim2, patience, batch_size, dropout_rate,log=None, early_stop_metric='acc'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== DataLoader ====
    Xtr_tensor = torch.tensor(X_train, dtype=torch.float32)
    ytr_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(Xtr_tensor, ytr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    # ==== Model ====
    model = GroupSparseDFS(
        X_train.shape[1], hidden_dim1, hidden_dim2, n_classes,
        per_feature_group_ids, dropout_rate=dropout_rate
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ====== No-early-stop Path ======
    if early_stop_metric.lower() == "none":
        if log:
            log(f"[INFO] Early stop disabled (early_stop_metric='none').")

        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                opt.zero_grad()
                logits = model(xb)
                ce_loss = criterion(logits, yb)

                # regularization
                l1_reg = torch.norm(model.feature_mask, 1)
                group_loss = sum(
                    torch.norm(model.feature_mask[model.groups == g], 2)
                    for g in torch.unique(model.groups)
                )
                loss = ce_loss + l1_strength * l1_reg + group_strength * group_loss
                loss.backward()
                opt.step()

                total_loss.append(loss.item())

            if log and (epoch % 50 == 0 or epoch == 1):
                log(f"[Full-Train] Epoch {epoch}/{n_epochs}  Loss={np.mean(total_loss):.4f}")

        model.eval()
        with torch.no_grad():
            logits_te = model(Xte)
            probs_te = torch.softmax(logits_te, dim=1).cpu().numpy()
            preds_te = np.argmax(probs_te, axis=1)

        return model, preds_te, probs_te, None, None

    # ====== Early-stopping Path ======
    best_metric = float("inf") if early_stop_metric.lower() == "loss" else -1
    best_model_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            logits = model(xb)
            ce_loss = criterion(logits, yb)

            l1_reg = torch.norm(model.feature_mask, 1)
            group_loss = sum(
                torch.norm(model.feature_mask[model.groups == g], 2)
                for g in torch.unique(model.groups)
            )
            loss = ce_loss + l1_strength * l1_reg + group_strength * group_loss
            loss.backward()
            opt.step()

        # ===== Validation =====
        model.eval()
        with torch.no_grad():
            logits_val = model(Xte)
            preds_val = torch.argmax(logits_val, dim=1)
            acc_val = accuracy_score(yte.cpu(), preds_val.cpu())
            bacc_val = balanced_accuracy_score(yte.cpu(), preds_val.cpu())
            loss_val = criterion(logits_val, yte).item()

        # Select metric
        if early_stop_metric.lower() == "loss":
            metric = loss_val
        elif early_stop_metric.lower() == "bacc":
            metric = bacc_val
        else:
            metric = acc_val
        metric = acc_val

        # ====== Detect Improvement ======
        improved = (
            metric < best_metric - 1e-6 if early_stop_metric.lower() == "loss"
            else metric > best_metric + 1e-6
        )

        if improved:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0
            best_model_state = {
                k: v.clone().detach().cpu()
                for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1

        if log and (epoch % 20 == 0 or epoch == 1):
            log(
                f"[Epoch {epoch:4d}] V_loss={loss_val:.4f} | V_acc={acc_val:.4f} | "
                f"V_BAcc={bacc_val:.4f} | Best={best_metric:.4f} | NoImprove={no_improve}"
            )

        if no_improve >= patience:
            if log:
                log(
                    f"Early-stop at epoch {epoch}, best epoch={best_epoch}, "
                    f"best {early_stop_metric}={best_metric:.4f}"
                )
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        logits_te = model(Xte)
        probs_te = torch.softmax(logits_te, dim=1).cpu().numpy()
        preds_te = np.argmax(probs_te, axis=1)

    return model, preds_te, probs_te, best_metric, best_epoch


def evaluate_dfs_cv(
    X, y, per_feature_group_ids, n_classes, lr, out_dir,
    n_epochs, hidden_dim1, hidden_dim2, patience,
    l1_strength, group_strength, batch_size, dropout_rate, log, early_stop_metric="loss"
):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "acc": [], "bacc": [], "f1m": [], "f1w": [],
        "auc": [], "mcc": [], "best_metric": [], "best_epoch": []
    }

    fold = 0
    for tr, te in skf.split(X, y):
        fold += 1
        log(f"==== Fold {fold} ====")

        # ====== Retrieve Best Model ======
        model, preds, probs, best_metric, best_epoch = dfs_train_eval(
            X[tr], y[tr], X[te], y[te],
            per_feature_group_ids, n_classes,
            n_epochs=n_epochs, lr=lr,
            l1_strength=l1_strength, group_strength=group_strength,
            hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
            patience=patience, batch_size=batch_size, dropout_rate=dropout_rate,
            early_stop_metric=early_stop_metric, 
            log=log
        )

        acc  = accuracy_score(y[te], preds)
        bacc = balanced_accuracy_score(y[te], preds)
        f1m  = f1_score(y[te], preds, average='macro')
        f1w  = f1_score(y[te], preds, average='weighted')
        mcc  = matthews_corrcoef(y[te], preds)

        try:
            auc = roc_auc_score(
                label_binarize(y[te], classes=np.arange(n_classes)),
                probs, average='macro', multi_class='ovr'
            )
        except Exception:
            auc = np.nan

        # ====== Logging =======
        log(f"Fold {fold} BestEpoch = {best_epoch} | BestMetric({early_stop_metric}) = {best_metric:.4f}")
        log(f"Fold {fold} ACC={acc:.4f}, MCC={mcc:.4f}, AUC={auc:.4f}")

        # ====== Save Results ======
        metrics["acc"].append(acc)
        metrics["bacc"].append(bacc)
        metrics["f1m"].append(f1m)
        metrics["f1w"].append(f1w)
        metrics["auc"].append(auc)
        metrics["mcc"].append(mcc)
        metrics["best_metric"].append(best_metric)
        metrics["best_epoch"].append(best_epoch)

        pd.DataFrame({"true": y[te], "pred": preds}).to_csv(
            os.path.join(out_dir, f"fold{fold}_pred.csv"), index=False
        )
        cm = confusion_matrix(y[te], preds)
        pd.DataFrame(cm).to_csv(
            os.path.join(out_dir, f"confusion_matrix_fold{fold}.csv"), index=False
        )


    cv_results_df = pd.DataFrame(metrics)
    fold_names = [f'fold_{i+1}' for i in range(len(metrics["acc"]))]
    cv_results_df.insert(0, 'fold', fold_names)
    cv_results_df.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)
    log(f"Saved per-fold results to cv_summary.csv")
    summary_mean = {k: float(np.nanmean(v)) for k, v in metrics.items()}
    pd.DataFrame([summary_mean]).to_csv(os.path.join(out_dir, "cv_summary_MEAN.csv"), index=False)
    log(f"CV Summary (Mean): {json.dumps(summary_mean, ensure_ascii=False)}")


    return summary_mean

def main():
    # 数据路径
    data_dir = "./"
    net_file = os.path.join(data_dir, "optimized_network_features_normalized.csv")
    omics_file = os.path.join(data_dir, "Multi-Omics_Data.csv")
    labels_file = os.path.join(data_dir, "label.csv")

    l1_strengths=0.1
    group_strengths=0.01
    
    learning_rates = 0.01
    patience = 100 
    metrics_to_monitor = "ACC"

    n_epochs = 1000
    hidden_dim1 = 512
    hidden_dim2 = 256
    dropout_rate = 0.7
    batch_size = 64

    base_out = os.path.join(data_dir, "results_multiclass_DFS_raw_v2")
    os.makedirs(base_out, exist_ok=True)
    temp_log = make_logger(base_out)


    temp_log("=== Step 0: Load data ===")
    X_net_df = load_table(net_file, temp_log)
    X_omics_df = load_table(omics_file, temp_log)
    labels_df = load_table(labels_file, temp_log)
    labels_series = labels_df.iloc[:, 0]

    common = X_net_df.index.intersection(X_omics_df.index).intersection(labels_series.index)
    X_net_df, X_omics_df, labels_series = X_net_df.loc[common], X_omics_df.loc[common], labels_series.loc[common]

    le = LabelEncoder(); y = le.fit_transform(labels_series.values); n_classes = len(np.unique(y))
    temp_log(f"Classes: {list(le.classes_)} | n_classes={n_classes}")

    scaler_net, scaler_om = StandardScaler(), StandardScaler()
    X_net_scaled, X_omics_scaled = scaler_net.fit_transform(X_net_df), scaler_om.fit_transform(X_omics_df)
    s_net, s_om = compute_modal_score_raw(X_net_df, y, cv=5, log=temp_log), compute_modal_score_raw(X_omics_df, y, cv=5, log=temp_log)
    weights = np.array([s_net, s_om]) / (s_net + s_om + 1e-9) 
    X_all = np.hstack([X_net_scaled * weights[0], X_omics_scaled * weights[1]])
    X_all_scaled = StandardScaler().fit_transform(X_all)
    per_feature_group_ids = np.array([0]*X_net_df.shape[1] + [1]*X_omics_df.shape[1])
    feature_names = [f"net:{c}" for c in X_net_df.columns] + [f"om:{c}" for c in X_omics_df.columns]


    run_dir = os.path.join(base_out, f"lr_{learning_rates}_l1_{l1_strengths}_grp_{group_strengths}_{metrics_to_monitor}__dropout_{dropout_rate}_batch_{batch_size}_dim_{hidden_dim1}_{hidden_dim2}_pat{patience}")
    os.makedirs(run_dir, exist_ok=True)
    log = make_logger(run_dir)
    log(f"=== Run Start: lr={learning_rates}, l1={l1_strengths}, grp={group_strengths}, patience={patience}, metric={metrics_to_monitor} ===")
    evaluate_dfs_cv(
         X_all_scaled, y, per_feature_group_ids, n_classes,
         lr=learning_rates, out_dir=run_dir, n_epochs=n_epochs, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
         l1_strength=l1_strengths, group_strength=group_strengths, 
         early_stop_metric=metrics_to_monitor, patience=patience, 
         batch_size=batch_size, dropout_rate=dropout_rate, log=log)

    log(f"=== Retraining on Full Data ===")
    model, preds_all, probs_all, _, _ = dfs_train_eval(
         X_all_scaled, y, X_all_scaled, y,
         per_feature_group_ids, n_classes,
         n_epochs=n_epochs, lr=learning_rates,
         l1_strength=l1_strengths, group_strength=group_strengths,patience=patience,
         batch_size=batch_size, dropout_rate=dropout_rate,
         hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
         early_stop_metric="none", 
         log=log)

    mask = model.feature_mask.detach().cpu().numpy()
    importance = np.abs(mask)
    pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
         "importance", ascending=False).to_csv(os.path.join(run_dir, "selected_features.csv"), index=False)
    log(f"=== Run Complete ===")

if __name__ == "__main__":
    main()