import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from data_factory.data_loader import SMDSegLoader
from sklearn.metrics import accuracy_score
from datetime import datetime

writer = SummaryWriter()

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            accuracy_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)
                preds = torch.argmax(output, dim=1) # for classification tasks you need to adapt this
                # acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                # accuracy_list.append(acc)
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                #accuracy = accuracy_score(y_true, y_pred)
                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                #labels = labels.to(self.device)

                # # Convert model output to predicted classes:
                # _, predictions = torch.max(output, dim=1)

                # # Move predictions and labels to CPU and convert to numpy if using sklearn:
                # y_true = labels.cpu().numpy()
                # y_pred = predictions.cpu().numpy()

                # # Now you can compute accuracy:
                # accuracy = accuracy_score(y_true, y_pred)
                # print("Accuracy:", accuracy)
                # #preds = output.argmax(dim=1)  # Assuming output is logits

# Update correct and total counts
 

 
                #writer.add_scalar('Train Accuracy', epoch_accuracy.item(), epoch * len(self.train_loader) + i)
                # from sklearn.metrics import accuracy_score
                # acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                writer.add_scalar('training loss', rec_loss.item() , epoch * len(self.train_loader) + i)
               # writer.add_scalar('Train Accuracy', acc, epoch * len(self.data_loader) + i)  
                print('epoch {}, loss_1 {}, loss_2 {},  rec_loss_ {}'.format(epoch * len(self.train_loader) + i  , loss1.item(), loss2.item(), rec_loss.item()))
        
           
           
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            #print(f"Epoch {epoch + 1}: Train Accuracy={epoch_accuracy:.4f}")
            train_loss = np.average(loss1_list)
            train_accuracy = np.average(accuracy_list)
            ####################################################################################################################TENSOR
            
            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, train_accuracy))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        writer.close()

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            

            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # Find all ground truth anomaly starts (regardless of predictions)
        anomaly_starts = np.where((gt[:-1] == 0) & (gt[1:] == 1))[0] + 1

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))



                # print('====================  GT values equal 1   ===================')
        # indices = np.where(gt == 1)[0]
        # print("Indices where gt is equal to 1:", ", ".join(map(str, indices)))
        ####################################################################################################
        #                                          Data Segment Extraction                                            #
        ####################################################################################################
        # Initialize the loader
        #data_path = "your/data/path"  # Replace with your actual data path
        loader = SMDSegLoader(self.data_path, win_size=100, step=10)

        # Access the TS variablefffff
        TS = loader.TS
       # print("Content of  TS:", TS[:100])

        #start_idx = np.random.choice(anomaly_starts)
        start_idx = 130000 # 43050
        segment_length = 10000
        def extract_random_segment(data, segment_length=10000, start_idx=130000, anomaly_starts=None):
            n = len(data)
            # (1) if data is too short, just return it all
            if n <= segment_length:
                return data

            # (2) decide where to start
            if start_idx is None:
                if anomaly_starts is not None and len(anomaly_starts) > 0:
                    start_idx = np.random.choice(anomaly_starts)
                else:
                    start_idx = np.random.randint(0, n - segment_length + 1)

            # (3) clamp to valid range
            start_idx = max(0, min(start_idx, n - segment_length))

            # (4) slice out the window
            return data[start_idx : start_idx + segment_length]

 
        
        print(f"start_idx: {start_idx}")
        as_segment = extract_random_segment(test_energy, segment_length, start_idx, anomaly_starts)
        gt_segment = extract_random_segment(gt, segment_length, start_idx, anomaly_starts) #ground truth
        TS_segment = extract_random_segment(TS, segment_length, start_idx, anomaly_starts) #Time Series Data
        pred_segment = extract_random_segment(pred, segment_length, start_idx, anomaly_starts)
 
        

        print('as segment shape', as_segment.shape)
        print(f"Anomaly Score values\n {as_segment}")
        #gt_segment=np.array(gt_segment) 
        print('gt segment shap', gt_segment.shape)########
        #print(f"gt values\n {gt_segment}")
        print(f"gt values\n\033[94m{gt_segment}\033[0m")


        ####################################################################################################
        #                                          Mat PLOT                                                 #
        ####################################################################################################
        
        import matplotlib.pyplot as plt
        import statistics

        def smooth(y, box_pts=3):
            box = np.ones(box_pts)/box_pts
            if len(y.shape) == 1:
                # 1D array - apply convolution directly
                y_smooth = np.convolve(y, box, mode='same')
            else:
                # Multi-dimensional array - apply convolution to each column
                y_smooth = np.zeros_like(y)
                for i in range(y.shape[1]):
                    y_smooth[:, i] = np.convolve(y[:, i], box, mode='same')
            return y_smooth
        
        # Use default matplotlib style to avoid issues with missing scienceplots
        plt.rcParams["text.usetex"] = False
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
        
        
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        plt.plot(smooth(TS_segment), label="Time Series Data", color='black', linewidth=1)
        #plt.title("Time Series Plot")
        plt.xlabel("Time")
        plt.ylabel("Value")
        #plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1 = plt.gca()  # Get the current axes (first subplot)
        ax1.tick_params(axis='both', direction='in')  # Set tick direction for both x and y axes
        ymin, ymax = plt.ylim()
        #plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
        plt.plot(as_segment, label='Anomaly Scores', color='blue', linewidth=1)
        plt.axhline(y=thresh, color='red', linestyle='--', label='Threshold', linewidth=0.3)
        plt.fill_between(range(len(as_segment)), 0, 1, where=(gt_segment == 1), color='blue', alpha=0.2, label='Ground Truth')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.ylim(0, 1)  # Set Y-axis range between 0 and 1 for anomaly scores
        #plt.title(f'Anomaly Scores Over Time (Area{start_idx})')
        #plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)  # <-- Legend on the left
        # Adjust tick direction for the second subplot
        ax2 = plt.gca()  # Get the current axes (second subplot)
        ax2.tick_params(axis='both', direction='in')  # Set tick direction for both x and y axes
 
        plt.tight_layout()

        # Save the combined plot to a file
        timestamp = datetime.now().strftime('%d%m%y%H%M')
        plot_filename = f'AAAAcombined_plot_idx_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {plot_filename}")
        



        # Additional multi-dimensional plotting if data has multiple dimensions
        # if len(TS_segment.shape) > 1 and TS_segment.shape[1] > 1:
        #     import matplotlib.pyplot as plt
        #     from matplotlib.backends.backend_pdf import PdfPages
            
        #     name = f"SMD_dim_analysis_{timestamp}"
        #     os.makedirs(os.path.join('plots', name), exist_ok=True)
        #     pdf = PdfPages(f'plots/{name}/output.pdf')
            
        #     for dim in range(TS_segment.shape[1]):
        #         y_t = TS_segment[:, dim]
        #         labels = gt_segment
        #         a_s = as_segment if len(as_segment.shape) == 1 else as_segment[:, dim]
                
        #         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #         ax1.set_ylabel('Value')
        #         ax1.set_title(f'Dimension = {dim}')
        #         ax1.plot(smooth(y_t), linewidth=0.5, label='Time Series', color='black')
                
        #         ax3 = ax1.twinx()
        #         ax3.plot(labels, '--', linewidth=0.3, alpha=0.5, color='red', label='Ground Truth')
        #         ax3.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.3)
                
        #         if dim == 0: 
        #             ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        #             ax3.legend(ncol=1, bbox_to_anchor=(0.6, 0.9))
                
        #         ax2.plot(smooth(a_s), linewidth=0.5, color='g', label='Anomaly Score')
        #         ax2.axhline(y=thresh, color='red', linestyle='--', linewidth=0.3, label='Threshold')
        #         ax2.set_xlabel('Timestamp')
        #         ax2.set_ylabel('Anomaly Score')
        #         ax2.legend()
                
        #         pdf.savefig(fig)
        #         plt.close()
        #     pdf.close()
        #     print(f"Multi-dimensional plots saved to plots/{name}/output.pdf")
        
        #plt.show()
        return accuracy, precision, recall, f_score

#writer.close()
        
        # # Save paired data: (time_index, anomaly_score, time_series_value)
        # paired_data = []
        # for i in range(len(as_segment)):
        #     time_index = start_idx + i
        #     anomaly_score = as_segment[i]
        #     ts_value = TS_segment[i]
        #     paired_data.append((time_index, anomaly_score, ts_value))
        
        # # Save to file with timestamp
        # timestamp = datetime.now().strftime('%d%m%y%H%M')
        # data_filename = f'paired_data_idx_{start_idx}_{timestamp}.txt'
        # latex_filename = f'latex_data_idx_{start_idx}_{timestamp}.tex'
        
        # # Save regular format
        # with open(data_filename, 'w') as f:
        #     f.write("# Format: (time_index, anomaly_score, time_series_values...)\n")
        #     f.write(f"# Segment range: {start_idx} to {start_idx + segment_length - 1}\n")
        #     f.write(f"# Threshold: {thresh}\n")
        #     f.write(f"# Time series dimensions: {TS_segment.shape[1] if len(TS_segment.shape) > 1 else 1}\n")
            
        #     for time_idx, as_val, ts_val in paired_data:
        #         # Handle both 1D and multi-dimensional time series
        #         if isinstance(ts_val, np.ndarray) and ts_val.ndim > 0:
        #             ts_str = " ".join([f"{val:.6f}" for val in ts_val])
        #             f.write(f"({time_idx}, {as_val:.6f}, [{ts_str}])\n")
        #         else:
        #             f.write(f"({time_idx}, {as_val:.6f}, {ts_val:.6f})\n")
        
        # # Save LaTeX-compatible format
        # with open(latex_filename, 'w') as f:
        #     f.write("% LaTeX pgfplots coordinates format\n")
        #     f.write(f"% Segment range: {start_idx} to {start_idx + segment_length - 1}\n")
        #     f.write(f"% Threshold: {thresh:.6f}\n\n")
            
        #     # Anomaly Scores plot
        #     f.write("% Anomaly Scores coordinates:\n")
        #     f.write("\\addplot+[smooth,tension=0.2, mark=o, solid, color=blue] coordinates {\n")
        #     for i, (time_idx, as_val, ts_val) in enumerate(paired_data):
        #         if i % 10 == 0:  # Sample every 10th point to avoid overcrowding
        #             f.write(f"\t({time_idx},{as_val:.6f})")
        #             if i < len(paired_data) - 10:
        #                 f.write(" ")
        #             if (i // 10 + 1) % 5 == 0:  # New line every 5 coordinates
        #                 f.write("\n")
        #     f.write("\n};\n\n")
            
        #     # Time Series plot - handle multi-dimensional data
        #     if len(TS_segment.shape) > 1 and TS_segment.shape[1] > 1:
        #         # Multi-dimensional: create plots for each dimension
        #         for dim in range(TS_segment.shape[1]):
        #             f.write(f"% Time Series coordinates (Dimension {dim}):\n")
        #             f.write(f"\\addplot+[smooth,tension=0.2, mark=none, solid, color=black] coordinates {{\n")
        #             for i, (time_idx, as_val, ts_val) in enumerate(paired_data):
        #                 if i % 10 == 0:  # Sample every 10th point to avoid overcrowding
        #                     val = ts_val[dim] if isinstance(ts_val, np.ndarray) else ts_val
        #                     f.write(f"\t({time_idx},{val:.6f})")
        #                     if i < len(paired_data) - 10:
        #                         f.write(" ")
        #                     if (i // 10 + 1) % 5 == 0:  # New line every 5 coordinates
        #                         f.write("\n")
        #             f.write("\n};\n\n")
        #     else:
        #         # Single dimension
        #         f.write("% Time Series coordinates:\n")
        #         f.write("\\addplot+[smooth,tension=0.2, mark=none, solid, color=black] coordinates {\n")
        #         for i, (time_idx, as_val, ts_val) in enumerate(paired_data):
        #             if i % 10 == 0:  # Sample every 10th point to avoid overcrowding
        #                 val = ts_val[0] if isinstance(ts_val, np.ndarray) else ts_val
        #                 f.write(f"\t({time_idx},{val:.6f})")
        #                 if i < len(paired_data) - 10:
        #                     f.write(" ")
        #                 if (i // 10 + 1) % 5 == 0:  # New line every 5 coordinates
        #                     f.write("\n")
        #         f.write("\n};\n\n")
            
        #     # Threshold line
        #     f.write("% Threshold line:\n")
        #     f.write(f"\\addplot+[no marks, dashed, color=red] coordinates {{\n")
        #     f.write(f"\t({start_idx},{thresh:.6f}) ({start_idx + segment_length - 1},{thresh:.6f})\n")
        #     f.write("};\n")
        
        # print(f"Paired data saved to {data_filename}")
        # print(f"LaTeX data saved to {latex_filename}")
        # print(f"Data range: index {start_idx} to {start_idx + len(as_segment) - 1}")
 