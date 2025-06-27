#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import matplotlib.pyplot as plt
import time
from util import *
import pandas as pd 
from datetime import timedelta
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

#pip install XlsxWriter
#jupyter nbconvert --to script Trainer.ipynb


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.sign = 1
        else:  # 'max'
            self.sign = -1

    def step(self, score):
        score = self.sign * score  # transforma max em min se necessário

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# In[ ]:


def compute_segmentation_metrics(preds, targets, num_classes, eps=1e-6):
    preds = preds.view(-1).cpu()
    targets = targets.view(-1).cpu()

    dice_total          = 0.0
    miou_total          = 0.0
    iou_valid_classes   = 0
    precision_per_class = []
    recall_per_class    = []

    classes_to_eval = [1] if num_classes == 1 else range(num_classes)

    for cls in classes_to_eval:
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        union = pred_sum + target_sum

        # Dice
        if union > 0:
            dice = (2.0 * intersection + eps) / (union + eps)
            dice_total += dice.item()

        # IoU
        union_iou = (pred_mask | target_mask).sum().float()
        if union_iou > 0:
            iou = (intersection + eps) / (union_iou + eps)
            miou_total += iou.item()
            iou_valid_classes += 1

        # Precision, Recall
        tp = intersection.item()
        fp = (pred_mask & ~target_mask).sum().float().item()
        fn = (~pred_mask & target_mask).sum().float().item()

        if (tp + fp + fn) > 0:
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)

            precision_per_class.append(precision)
            recall_per_class.append(recall)

    mean_dice = dice_total / len(classes_to_eval)
    mean_iou = miou_total / iou_valid_classes if iou_valid_classes > 0 else 0.0
    mean_precision = np.mean(precision_per_class) if precision_per_class else 0.0
    mean_recall = np.mean(recall_per_class) if recall_per_class else 0.0
    q = mean_iou * mean_dice

    #DICE = F1
    return mean_dice, mean_iou, mean_precision, mean_recall, q


# faz o cálculo imagem por imagem e depois tira a média.
def compute_iou(preds, masks, num_classes=1, eps=1e-6):
    iou_per_class = [ [] for _ in range(num_classes) ]  # lista de listas

    batch_size = preds.size(0)

    for i in range(batch_size):
        pred = preds[i]
        mask = masks[i]

        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            mask_cls = (mask == cls).float()

            if mask_cls.sum() == 0:
                continue  # não avalia classe ausente no ground truth

            intersection = (pred_cls * mask_cls).sum()
            union = ((pred_cls + mask_cls) > 0).float().sum()
            iou = (intersection + eps) / (union + eps)
            iou_per_class[cls].append(iou)

    # média por classe
    class_ious = [
        torch.stack(iou_list).mean()
        for iou_list in iou_per_class
        if len(iou_list) > 0
    ]

    if not class_ious:
        return 0.0

    return torch.stack(class_ious).mean().item()




# In[ ]:


class Trainer:

    model         = None
    criterion     = None
    optimizer     = None
    scheduler     = None
    learning_rate = None

    #Essa classe treina apenas segmentacao com 1 classe
    #Se necessario mais, usar SemanticTrainer
    num_classes   = 2

    def __init__(self, model_filename=None, model_dir=None, info={}, save_xlsx=False):

        if save_xlsx:
            if model_filename is None:
                raise Exception("model_filename é obrigatório ao com save_xlsx == True")
        self.save_xlsx = save_xlsx

        #salva o nome e diretorio do modelo
        self.model_filename = model_filename
        if model_dir is None:
            model_dir = model_filename
        self.model_dir = model_dir

        #se pelo menos o nome do modelo for passado
        if self.model_filename is not None:
            self.model_file_dir = self.model_dir + "/" + self.model_filename
        else:
            self.model_file_dir = None


        #informacoes extra a serem salvas no xslx
        self.info = info
        #index da imagem sample que sera usada
        #para salvar o output durante o treino
        self.sample_img_fixed_index = 0

    def create_criterion(self):
        self.info['loss_function'] = 'BCEWithLogitsLoss'
        self.criterion = nn.BCEWithLogitsLoss()

    def create_scheduler(self, patience=10, factor=0.5, mode='max'):
        self.info['scheduler'] = "ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, verbose=True)"
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                      mode=mode, 
                                      patience=patience, 
                                      factor=factor)



    def create_optimizer(self):
        self.info['optimizer'] = f"optim.Adam(self.model.parameters(), lr={self.learning_rate})"
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def get_model_output(self,images):
        return self.model(images)


    def train_loop(self, images, masks, epoch):
        outputs     = self.get_model_output(images)

        outputs_s   = outputs.squeeze(1)
        masks_s     = masks.squeeze(1).float()

        loss    = self.criterion(outputs_s, masks_s)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss = loss.item() * images.size(0)

        return train_loss




    def val_loop(self, images, masks):
        outputs     = self.get_model_output(images)

        loss        = self.criterion(outputs, masks)
        val_loss    = loss.item() * images.size(0)


        #faz o threshold
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).long()
        masks = masks.long()


        #computa as metricas
        dice, mIoU, precision, recall, q = compute_segmentation_metrics(preds, masks, num_classes=self.num_classes)
        IoU = compute_iou(preds, masks, num_classes=self.num_classes)

        val_dice      = dice      * images.size(0)
        val_mIoU      = mIoU      * images.size(0)
        val_IoU       = IoU       * images.size(0)
        val_precision = precision * images.size(0)
        val_recall    = recall    * images.size(0)
        val_q         = q         * images.size(0)

        return val_loss, val_dice, val_mIoU, val_IoU, val_precision, val_recall, val_q



    def update_history(self, history, train_loss=None, loss=None, dice=None, miou=None,
                   iou=None, precision=None, recall=None, q=None,
                   elapsed_time=None, images_per_sec=None, started=None):
        if train_loss is not None:
            history["train_loss"].append(train_loss)
        if loss is not None:
            history["loss"].append(loss)
        if dice is not None:
            history["dice"].append(dice)
        if miou is not None:
            history["miou"].append(miou)
        if iou is not None:
            history["iou"].append(iou)
        if precision is not None:
            history["precision"].append(precision)
        if recall is not None:
            history["recall"].append(recall)
        if q is not None:
            history["q"].append(q)
        if elapsed_time is not None:
            history["elapsed_time"].append(elapsed_time)
        if images_per_sec is not None:
            history["images_per_sec"].append(images_per_sec)
        if started is not None:
            history["started"].append(started)



    def train(self, model, 
                    train_loader, 
                    val_loader,
                    test_loader, 
                    num_epochs=50, 
                    #salva o modelo a cada
                    save_every=None,
                    #imprime o andamento a cada
                    print_every=None,
                    #continua o treinamento de onde parou
                    continue_from_last=False,
                    #salva a saida da rede a cada
                    save_outputs_every=None,
                    #verbose==1 imprime o treinamento na mesma linha
                    verbose=3,
                    learning_rate=1e-4,
                    # apos essa quantidade de epocas o treino
                    # se a acuracia diminuir o treino ira voltar para o -best
                    #e tentar novamente, até que aumente ou exceda a quantidade de tentativas
                    scheduler_patience=10,
                    early_stop_patience=20
                    ):

        torch.backends.cudnn.benchmark = True
        device = self.get_device()

        self.learning_rate  = learning_rate
        self.model          = model
        start_epoch         = 0
        best_score          = -1.0
        best_stats          = ""
        start_time          = time.time()
        started             = False
        best_path           = self.model_file_dir.replace('.pth', '-best.pth')
        last_path           = self.model_file_dir.replace('.pth', '-last.pth')
        batch_size          = train_loader.batch_size

        trainable_parameters = count_trainable_parameters(model)
        print("trainable_parameters:", trainable_parameters)
        self.info['dataset_name']         = train_loader.dataset.__module__
        self.info['dataset_batch_size']   = batch_size
        self.info['trainable_parameters'] = trainable_parameters
        images, labels = next(iter(train_loader))
        self.info['dataset_resolution']   = f"{images.shape[2]} x {images.shape[3]}"


        self.val_history = {
            "train_loss":     [],
            "loss":           [],
            "dice":           [],
            "miou":           [],
            "iou":            [],
            "precision":      [],
            "recall":         [],
            "q":              [],
            "elapsed_time":   [],
            "images_per_sec": [],
            "started":        [],
        }
        self.test_history = {k: [] for k in self.val_history}


        #imprime tudo na mesma linha
        tqdm_disable = print_every!=None
        print_end    = '\r\n'
        if verbose == 1:
            print_end    = '\r'
            tqdm_disable = True



        #se o nome do modelo foi passado
        if self.model_filename is not None:
            #cria os diretorios
            os.makedirs(self.model_dir, exist_ok=True)

            #primeiro, verifica se o modelo final, treinado ja existe
            if os.path.exists(self.model_file_dir):
                #se ja existir, carrega e retorna
                print("Modelo treinado já existe.")
                self.load_model(self.model_file_dir)
                self.print_last_history_stats()
                return model
            #se nao existir e for uma continuacao do treinamento
            elif continue_from_last == True:
                #continua a partir do -last
                if os.path.exists(last_path):
                    _, _, start_epoch, start_time = self.load_model(last_path)
                    print(f"Continuando do modelo salvo: {last_path}")
                    print(f"start_epoch: {start_epoch}, start_time: {start_time}")
                    if start_epoch >= num_epochs:
                        self.print_last_history_stats()
                        return self.model



        model.to(device)
        self.create_criterion()
        self.create_optimizer()
        self.create_scheduler(patience=scheduler_patience)
        early_stopper = EarlyStopping(patience=early_stop_patience, mode='max')


        # Se a opcao de salvar output a cada epoca for enviado
        # Pega uma imagem fixa fora do loop de treinamento
        if save_outputs_every is not None:
            sample_imgs, sample_masks = next(iter(val_loader))
            self.sample_img_fixed  = sample_imgs[0+self.sample_img_fixed_index:self.sample_img_fixed_index+1].to(device)
            self.sample_mask_fixed = sample_masks[0+self.sample_img_fixed_index:self.sample_img_fixed_index+1]


        ## Treinamento
        epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):

            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Treinando Epoch {epoch+1}", disable=tqdm_disable):
                images = images.to(device)
                masks  = masks.to(device)
                ## loop de treino
                train_loss += self.train_loop(images, masks, epoch)

            avg_train_loss = train_loss / len(train_loader.dataset)


            ## Validação
            model.eval()
            val_loss        = 0.0
            val_dice        = 0.0
            val_mIoU        = 0.0
            val_IoU         = 0.0
            val_precision   = 0.0
            val_recall      = 0.0
            val_q           = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks  = masks.to(device)
                    #loop de validacao
                    loss, dice, mIoU, IoU, precision, recall, q = self.val_loop(images, masks)


                    val_loss      += loss
                    val_dice      += dice
                    val_mIoU      += mIoU
                    val_IoU       += IoU
                    val_precision += precision
                    val_recall    += recall
                    val_q         += q

            avg_val_loss        = val_loss / len(val_loader.dataset)
            avg_val_dice        = val_dice / len(val_loader.dataset)
            avg_val_mIoU        = val_mIoU / len(val_loader.dataset)
            avg_val_IoU         = val_IoU  / len(val_loader.dataset)
            avg_val_precision   = val_precision   / len(val_loader.dataset)
            avg_val_recall      = val_recall   / len(val_loader.dataset)
            avg_val_q           = val_q    / len(val_loader.dataset)


            ## Test
            test_loss        = 0.0
            test_dice        = 0.0
            test_mIoU        = 0.0
            test_IoU         = 0.0
            test_precision   = 0.0
            test_recall      = 0.0
            test_q           = 0.0
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks  = masks.to(device)
                    loss, dice, mIoU, IoU, precision, recall, q = self.val_loop(images, masks)

                    test_loss      += loss
                    test_dice      += dice
                    test_mIoU      += mIoU
                    test_IoU       += IoU
                    test_precision += precision
                    test_recall    += recall
                    test_q         += q

            avg_test_loss        = test_loss / len(test_loader.dataset)
            avg_test_dice        = test_dice / len(test_loader.dataset)
            avg_test_mIoU        = test_mIoU / len(test_loader.dataset)
            avg_test_IoU         = test_IoU  / len(test_loader.dataset)
            avg_test_precision   = test_precision   / len(test_loader.dataset)
            avg_test_recall      = test_recall   / len(test_loader.dataset)
            avg_test_q           = test_q    / len(test_loader.dataset)

            elapsed     = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            current_lr  = self.optimizer.param_groups[0]['lr']
            stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                    f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                    f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} IoU: {avg_val_IoU:.4f} " 
                    f"Precision: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                    f"Tempo total: {elapsed_str} LR:{current_lr:.6f}")


            if print_every is None:
                print(stats,end=print_end)
            else:
                if (epoch+1) % print_every == 0:
                    print(stats,end=print_end)


            images_per_sec = (len(train_loader) * batch_size) / elapsed

            ## Salva a evolucao da rede
            self.update_history(
                self.val_history,
                train_loss=avg_train_loss,
                loss=avg_val_loss,
                dice=avg_val_dice,
                miou=avg_val_mIoU,
                iou=avg_val_IoU,
                precision=avg_val_precision,
                recall=avg_val_recall,
                q=avg_val_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )
            self.update_history(
                self.test_history,
                train_loss=avg_train_loss,
                loss=avg_test_loss,
                dice=avg_test_dice,
                miou=avg_test_mIoU,
                iou=avg_test_IoU,
                precision=avg_test_precision,
                recall=avg_test_recall,
                q=avg_test_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )




            started = True


            # O avg_val_dice será observado para o scheduler e early_stopper

            # reduz o learning rate caso o score nao melhore
            self.scheduler.step(avg_val_dice)

            # para o treinamento caso nao melhore em X epocas
            early_stopper.step(avg_val_dice)
            if early_stopper.early_stop:
                print(f"Parando na época {epoch+1} por early stopping.")
                break

            ## Salva o melhor modelo ate o momento
            if avg_val_dice > best_score:
                best_score = avg_val_dice

                if self.model_file_dir is not None:
                    #salva o modelo na melhor epoca
                    self.save_model(best_path, epoch, best_score)
                    current_lr  = self.optimizer.param_groups[0]['lr']
                    best_stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                                f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                                f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} IoU: {avg_val_IoU:.4f} " 
                                f"Precision: {avg_val_precision:.4f} " 
                                f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                                f"Tempo total: {elapsed_str} LR:{current_lr:.6f}")
                    if print_every is None and verbose > 1:
                        print("✔ Melhor modelo salvo:", best_stats, end=print_end)
                    #salva o excel ate o momento atual
                    if self.save_xlsx:
                        self.do_save_xlsx()





            #Se for para salvar a evolucao do predict da rede
            ## Salvar saída da rede em imagem
            if save_outputs_every is not None and (epoch + 1) % save_outputs_every == 0:
                model.eval()
                with torch.no_grad():
                    fig, axs = self._save_output(epoch=epoch+1)
                    plt.tight_layout()
                    outdir = os.path.join(self.model_dir, 'outputs') if self.model_dir else 'outputs'
                    os.makedirs(outdir, exist_ok=True)
                    plt.savefig(os.path.join(outdir, f"{epoch+1:03d}.png"))
                    plt.close()

            if save_every is not None and (epoch + 1) % save_every == 0:
                last_model_file_dir = self.model_file_dir.replace('.pth','-last.pth')
                self.save_model(last_model_file_dir, epoch, best_score)
                self.do_save_xlsx()
                if verbose > 1:
                    print("Saved last as", last_model_file_dir, end=print_end)



        last_stats = (f"Epoch [{epoch+1}/{num_epochs}] - " 
                    f"Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} " 
                    f"Dice: {avg_val_dice:.4f} mIoU: {avg_val_mIoU:.4f} IoU: {avg_val_IoU:.4f} " 
                    f"Precision: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} " 
                    f"Tempo total: {elapsed_str} LR:{current_lr:.6f}")



        #calcula o FPS do modelo
        self.info['FPS'], self.info['time_per_image'] = measure_inference_speed(self.model, val_loader)

        print("")
        if best_stats:
            print("Melhor modelo:\r\n", best_stats)
        print("Ultimo modelo:\r\n", last_stats + ' FPS:',self.info['FPS'])


        if self.model_file_dir is not None:
            self.save_model(self.model_file_dir, epoch, best_score)
            print("Saved as", self.model_file_dir)


        if self.save_xlsx:
            # Escreve o arquivo excel com o history
            self.do_save_xlsx()

        #beep win
        #os.system('powershell.exe -Command "[console]::beep(600,200); [console]::beep(600,200);"')
        #linux
        os.system('play -nq -t alsa synth 0.2 sine 600; play -nq -t alsa synth 0.2 sine 600')
        return model


    def print_last_history_stats(self):
        #INFO
        last_stats = ""
        keys = list(self.info.keys())
        for key in keys:
            value = self.info[key]
            last_stats += f"{key}:{value} "
        print(last_stats)
        #History
        last_stats = ""
        keys = list(self.val_history.keys())
        for key in keys:
            value = self.val_history[key][-1]
            last_stats += f"{key}:{value} "
        print(last_stats)



    def _save_output(self,epoch=None):
        with torch.no_grad():
            output = self.get_model_output(self.sample_img_fixed)

            # Aplicar sigmoid e limiar
            out_sigmoid = torch.sigmoid(output[0])
            out_thresh = (out_sigmoid > 0.5).float()
            out_thresh_np = out_thresh.cpu().squeeze().numpy()
            sample_mask_fixed_s = self.sample_mask_fixed.squeeze().cpu().numpy()

            # Inicializar imagem RGB de saída com zeros (preto = true negative)
            h, w = sample_mask_fixed_s.shape
            diff_img = np.zeros((h, w, 3), dtype=np.float32)

            # Máscaras para os diferentes tipos de pixel
            tp = (out_thresh_np == 1) & (sample_mask_fixed_s == 1)
            fn = (out_thresh_np == 0) & (sample_mask_fixed_s == 1)
            fp = (out_thresh_np == 1) & (sample_mask_fixed_s == 0)

            # Aplicar cores
            diff_img[tp] = [1, 1, 1]       # Branco
            diff_img[fn] = [1, 0.5, 0]     # Laranja
            diff_img[fp] = [1, 0, 0]       # Vermelho

            # Criar a figura
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(sample_mask_fixed_s, cmap='gray')
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')

            axs[1].imshow(out_thresh_np, cmap='gray')
            axs[1].set_title("Predição")
            axs[1].axis('off')

            axs[2].imshow(diff_img)
            axs[2].set_title("Diferença (TP=branco, FN=laranja, FP=vermelho)")
            axs[2].axis('off')

            suptitle = self.model_filename.replace('.pth', '')
            if epoch is not None:
                suptitle += f" epoch:{epoch}"

            fig.suptitle(suptitle, fontsize=14)
            return fig, axs



    def save_sample_output(self, data_loader, samples=[0]):

        if self.model is None:
            raise Exception("O modelo ainda não foi carregado, use trainer.load_model(model=model) passando o objeto do modelo.")

        #define qual sera a amostra
        device                      = next(self.model.parameters()).device
        if type(samples) == int:
            samples = [samples]

        for sample in samples:
            sample_imgs, sample_masks   = next(iter(data_loader))
            self.sample_img_fixed       = sample_imgs[0+sample:sample+1].to(device)
            self.sample_mask_fixed      = sample_masks[0+sample:sample+1]
            #gera o output
            fig, axs = self._save_output()
            #salva a amostra
            plt.tight_layout()
            outdir = os.path.join(self.model_dir, 'outputs') if self.model_dir else 'outputs'
            os.makedirs(outdir, exist_ok=True)
            fig_path = os.path.join(outdir, f"sample{sample}-{self.model_filename}").replace('.pth','.png')
            plt.savefig(fig_path)
            plt.close()
            print(fig_path, 'saved.')



    def do_save_xlsx(self):

        avg_speed = sum(self.val_history['images_per_sec']) / len(self.val_history['images_per_sec'])
        self.info['training_speed_img_per_sec'] = round(avg_speed, 2)


        hist_name = self.model_file_dir.replace('.pth', '.xlsx')
        df_val_history = pd.DataFrame(self.val_history)
        df_val_history.insert(0, 'epoch', range(1, len(df_val_history)+1))
        df_val_history['epoch'] = df_val_history['epoch'].astype(str)


        df_test_history = pd.DataFrame(self.test_history)
        df_test_history.insert(0, 'epoch', range(1, len(df_test_history)+1))
        df_test_history['epoch'] = df_test_history['epoch'].astype(str)


        df_info = pd.DataFrame(self.info, index=[0])
        with pd.ExcelWriter(hist_name, engine='xlsxwriter') as writer:
            df_val_history.to_excel(writer, sheet_name='val_history', index=False, float_format="%.4f")
            df_test_history.to_excel(writer, sheet_name='test_history', index=False, float_format="%.4f")
            df_info.to_excel(writer, sheet_name='model_info', index=False, float_format="%.4f")

            workbook  = writer.book
            worksheet = writer.sheets['val_history']

            chart = workbook.add_chart({'type': 'line'})

            # A coluna 'epoch' agora está na coluna 0
            # Supondo que 'val_dice' esteja na coluna 5 e 'val_IoU' na 6 (ou ajuste isso dinamicamente)
            col_dice = df_val_history.columns.get_loc('dice')
            col_iou  = df_val_history.columns.get_loc('miou')

            chart.add_series({
                'name':       'dice',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],  # coluna 0 = epoch
                'values':     ['val_history', 1, col_dice, len(df_val_history), col_dice],
            })
            chart.add_series({
                'name':       'mIoU',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],
                'values':     ['val_history', 1, col_iou, len(df_val_history), col_iou],
            })

            chart.set_title({'name': 'Treinamento'})

            chart.set_x_axis({
                'name': 'Época',
                'interval_unit': 10,
                'num_font': {'rotation': -45},
            })
            chart.set_y_axis({'name': 'Valor'})

            worksheet.insert_chart('K2', chart)



    def load_xlsx_history(self):
        hist_name = self.model_file_dir.replace('.pth', '.xlsx').replace('-best','')

        # Lê todas as planilhas do arquivo
        xls = pd.read_excel(hist_name, sheet_name=None)

        # Recupera o DataFrame de histórico e converte para lista de dicionários
        df_val_history   = xls['val_history']
        last_epoch   = int(df_val_history['epoch'].iloc[-1])
        self.val_history = df_val_history.drop(columns=['epoch']).to_dict(orient='list')


        df_test_history   = xls['test_history']
        self.test_history = df_test_history.drop(columns=['epoch']).to_dict(orient='list')

        # Tempo acumulado
        elapsed_str      = df_val_history['elapsed_time'].iloc[-1]
        h, m, s          = map(int, elapsed_str.split(':'))
        accumulated_time = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
        start_time       = time.time() - accumulated_time  # Ajusta para manter contagem acumulada

        # Recupera o DataFrame de informações do modelo e converte para dicionário
        df_info = xls['model_info']
        self.info = df_info.iloc[0].to_dict()
        return last_epoch, start_time

    def load_model(self, model_file_dir, model=None, load_xlsx=True, load_scheduler=False):
        #se o modelo for passado
        if model is not None:
            #self.model recebe o novo modelo
            self.model = model
        #se o modelo a ser carregado nao foi passado
        if self.model is None:
            raise Exception("Voce precisa passar o objeto do modelo no parametro 'model'")

        if self.optimizer is None:
            self.create_optimizer()
        if self.scheduler is None:
            self.create_scheduler()

        #carrega o modelo do arquivo .pth
        checkpoint = torch.load(model_file_dir, weights_only=False)
        #recupera os states do arquivo
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_score  = checkpoint['best_acc']
        epoch       = checkpoint['epoch'] + 1
        self.model.to(self.get_device())
        print(f"Modelo carregado: {model_file_dir}")
        if load_xlsx:
            start_epoch, start_time = self.load_xlsx_history()
            return best_score, epoch, start_epoch, start_time
        return best_score, epoch

    def save_model(self, path, epoch, best_score):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_acc': best_score
                }, path)

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == '__main__':
    pass

