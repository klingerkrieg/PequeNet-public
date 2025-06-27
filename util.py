import torch
import time
import os
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import gc

def compute_iou(preds, masks, num_classes=1, threshold=0.5, eps=1e-6):
    """
    Calcula IoU entre preds e masks.
    Para binário, preds pode ser logits; para multiclasse, deve ser logits com shape [B, C, H, W]
    """
    if num_classes == 1:
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()
        masks = masks.float()
    else:
        preds = torch.argmax(preds, dim=1)  # [B, H, W]

    batch_size = preds.size(0)
    ious = []

    for i in range(batch_size):
        pred = preds[i]
        mask = masks[i]

        if num_classes == 1:
            if mask.sum() == 0:
                continue
            intersection = (pred * mask).sum()
            union = ((pred + mask) > 0).float().sum()
            iou = (intersection + eps) / (union + eps)
            ious.append(iou)
        else:
            iou_per_class = []
            for cls in range(num_classes):
                pred_cls = (pred == cls).float()
                mask_cls = (mask == cls).float()

                if mask_cls.sum() == 0:
                    continue

                intersection = (pred_cls * mask_cls).sum()
                union = ((pred_cls + mask_cls) > 0).float().sum()
                iou = (intersection + eps) / (union + eps)
                iou_per_class.append(iou)

            if iou_per_class:
                ious.append(torch.stack(iou_per_class).mean())

    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()



def compute_dice(preds, masks, num_classes=1, threshold=0.5, eps=1e-6):
    """
    Calcula Dice Score entre preds e masks.
    """
    if num_classes == 1:
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()
        masks = masks.float()

        intersection = (preds * masks).sum(dim=(1, 2, 3))
        total = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (total + eps)
        return dice.mean().item()
    else:
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        batch_size = preds.size(0)
        dices = []

        for i in range(batch_size):
            dice_per_class = []
            for cls in range(num_classes):
                pred_cls = (preds[i] == cls).float()
                mask_cls = (masks[i] == cls).float()

                if mask_cls.sum() == 0:
                    continue

                intersection = (pred_cls * mask_cls).sum()
                total = pred_cls.sum() + mask_cls.sum()
                dice = (2 * intersection + eps) / (total + eps)
                dice_per_class.append(dice)

            if dice_per_class:
                dices.append(torch.stack(dice_per_class).mean())

        if not dices:
            return 0.0
        return torch.stack(dices).mean().item()



def dice_loss(pred, target, num_classes=1, smooth=1.):
    """
    Calcula Dice Loss para binário ou multiclasse.
    """
    if num_classes == 1:
        # pred e target devem estar no formato [B, H, W]
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    else:
        pred = torch.softmax(pred, dim=1)  # [B, C, H, W]
        # One-hot encode target: [B, H, W] → [B, C, H, W]
        target = nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

    


def show_dataset_prev(train_loader, test_loader, val_loader=None, num_images=3, num_classes=1):
    images_shown = 0

    # Cria um iterador para o val_loader se for fornecido
    val_iter = iter(val_loader) if val_loader is not None else None

    for (images_train, masks_train), (images_test, masks_test) in zip(train_loader, test_loader):
        if val_iter:
            try:
                images_val, masks_val = next(val_iter)
            except StopIteration:
                break  # Encerra se o val_loader acabar

        for i in range(images_train.size(0)):
            if images_shown >= num_images:
                break

            def process_image_mask(img_tensor, mask_tensor):
                img_tensor = img_tensor.cpu()
                img = img_tensor.permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.squeeze(0).numpy()
                img = img * 0.5 + 0.5
                mask = mask_tensor.cpu().squeeze().numpy()
                return img, mask

            img_train, mask_train = process_image_mask(images_train[i], masks_train[i])
            img_test, mask_test = process_image_mask(images_test[i], masks_test[i])
            
            if val_iter:
                img_val, mask_val = process_image_mask(images_val[i], masks_val[i])

            # Define número de colunas com base no val_loader
            n_cols = 6 if val_iter else 4
            fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 4))

            axs[0].imshow(img_train)
            axs[0].set_title("Imagem Treino")
            axs[1].imshow(mask_train, cmap='viridis', vmin=0, vmax=num_classes)
            axs[1].set_title("Máscara Treino")

            axs[2].imshow(img_test)
            axs[2].set_title("Imagem Teste")
            axs[3].imshow(mask_test, cmap='viridis', vmin=0, vmax=num_classes)
            axs[3].set_title("Máscara Teste")

            if val_iter:
                axs[4].imshow(img_val)
                axs[4].set_title("Imagem Val")
                axs[5].imshow(mask_val, cmap='viridis', vmin=0, vmax=num_classes)
                axs[5].set_title("Máscara Val")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            images_shown += 1

        if images_shown >= num_images:
            break


def show_classification_dataset_prev(train_loader, test_loader, num_images=3):
    images_shown = 0
    for (images_train, labels_train), (images_test, labels_test) in zip(train_loader, test_loader):
        for i in range(images_train.size(0)):
            if images_shown >= num_images:
                break

            # Para o conjunto de treinamento
            img_train_tensor = images_train[i].cpu()
            if img_train_tensor.shape[0] == 3:  # Imagem RGB (C=3)
                img_train = img_train_tensor.permute(1, 2, 0).numpy()
            else:  # Imagem em escala de cinza (C=1)
                img_train = img_train_tensor.squeeze(0).numpy()
            label_train = labels_train[i]
            img_train = img_train * 0.5 + 0.5  # desfaz normalização

            # Para o conjunto de teste
            img_test_tensor = images_test[i].cpu()
            if img_test_tensor.shape[0] == 3:
                img_test = img_test_tensor.permute(1, 2, 0).numpy()
            else:
                img_test = img_test_tensor.squeeze(0).numpy()
            label_test = labels_test[i]
            img_test = img_test * 0.5 + 0.5  # desfaz normalização

            # Exibindo as 2 colunas: Imagem de treino, Máscara de treino, Imagem de teste, Máscara de teste
            fig, axs = plt.subplots(1, 2, figsize=(5, 2))
            
            # Treinamento
            axs[0].imshow(img_train, cmap='gray')
            axs[0].set_title(f"Treinamento class={label_train}")
            
            # Teste
            axs[1].imshow(img_test, cmap='gray')
            axs[1].set_title(f"Teste label={label_test}")

            # Ajustes para as imagens
            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            images_shown += 1
        if images_shown >= num_images:
            break
        
def show_predictions(models, data_loader, do_threshold=True, num_classes=1, max_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(models, list):
        models = [models]

    for model in models:
        model.to(device)
        model.eval()

    images_shown = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            for i in range(images.size(0)):
                if images_shown >= max_images:
                    break

                outputs = []
                for model in models:
                    output = model(images)  # (B, C, H, W)

                    if num_classes == 1:
                        pred = torch.sigmoid(output)
                        if do_threshold:
                            pred = (pred > 0.5).float()
                        pred = pred[i].cpu().squeeze().numpy()
                    else:
                        pred = torch.softmax(output, dim=1)
                        pred = torch.argmax(pred, dim=1)
                        pred = pred[i].cpu().numpy()

                    outputs.append(pred)

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * 0.5 + 0.5  # Reverte normalização

                mask = masks[i].cpu()
                if num_classes == 1:
                    mask = mask.squeeze().numpy()
                else:
                    if mask.ndim > 2:
                        mask = torch.argmax(mask, dim=0).cpu().numpy()
                    else:
                        mask = mask.cpu().numpy()

                # Plot
                fig, axs = plt.subplots(1, 2 + len(models), figsize=(12, 4))
                axs[0].imshow(img)
                axs[0].set_title("Imagem")
                axs[1].imshow(mask, cmap='tab10' if num_classes > 1 else 'gray', vmin=0, vmax=num_classes-1 if num_classes > 1 else None)
                axs[1].set_title("Máscara Real")

                for j, pred in enumerate(outputs):
                    axs[2 + j].imshow(pred, cmap='tab10' if num_classes > 1 else 'gray', vmin=0, vmax=num_classes-1 if num_classes > 1 else None)
                    axs[2 + j].set_title(f"Model {j + 1}")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                plt.show()

                images_shown += 1

            if images_shown >= max_images:
                break





def eval(model, test_loader, num_classes=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            if num_classes > 1:
                masks = masks.long()  # Importante! precisa ser long/int64

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            iou = compute_iou(outputs, masks, num_classes=num_classes)
            val_iou += iou * images.size(0)

            dice = compute_dice(outputs, masks, num_classes=num_classes)
            val_dice += dice * images.size(0)

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_iou = val_iou / len(test_loader.dataset)
    avg_val_dice = val_dice / len(test_loader.dataset)

    print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")


    
def verificar_mascara_multiclasse(mascara, num_classes):
    if len(mascara.shape) != 2:
        print("❌ Formato inválido: a máscara deve ser [H, W]")
    else:
        print("✅ Formato ok")

    valores = np.unique(mascara)
    if not np.all(np.equal(valores, valores.astype(int))):
        print("❌ A máscara contém valores não inteiros")
    else:
        print(f"✅ Valores únicos: {valores}")

    if mascara.dtype not in [np.uint8, np.int32, np.int64]:
        print(f"❌ Tipo inválido: {mascara.dtype}")
    else:
        print("✅ Tipo de dado ok")

    if valores.min() < 0 or valores.max() >= num_classes:
        print(f"❌ Valores fora do intervalo esperado [0, {num_classes - 1}]")
    else:
        print("✅ Intervalo de valores está correto")


def beep():
    os.system('powershell.exe -Command "[console]::beep(500,400); [console]::beep(500,400)"')


def count_trainable_parameters(model, format=False):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if format:
        return f"{total:,}".replace(",", ".")
    return total


def show_diff_mask(mask_tensor, output_tensor, threshold=0.5):
    """
    Exibe uma imagem com:
    - Verdadeiro Positivo (1,1): branco
    - Verdadeiro Negativo (0,0): preto
    - Falso Positivo (0,1): laranja
    - Falso Negativo (1,0): vermelho

    Parâmetros:
    - mask_tensor: tensor ground truth (shape [1, H, W] ou [H, W])
    - output_tensor: tensor do modelo (shape [1, H, W] ou [H, W]), antes ou depois da sigmoid
    - threshold: valor de corte para binarizar o output
    """

    # Preparar as máscaras binárias
    mask = mask_tensor.squeeze().cpu().numpy()
    output = output_tensor.squeeze().cpu().numpy()
    
    if output.dtype != np.bool_ and output.max() > 1:
        output = 1 / (1 + np.exp(-output))  # aplicar sigmoid se necessário

    mask_bin = (mask > 0.5).astype(np.uint8)
    output_bin = (output > threshold).astype(np.uint8)

    # Inicializar imagem RGB
    h, w = mask_bin.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Condições
    TP = (mask_bin == 1) & (output_bin == 1)
    TN = (mask_bin == 0) & (output_bin == 0)
    FN = (mask_bin == 1) & (output_bin == 0)
    FP = (mask_bin == 0) & (output_bin == 1)

    # Cores
    img[TP] = [255, 255, 255]   # branco
    img[TN] = [0, 0, 0]         # preto
    img[FP] = [255, 0, 0]       # vermelho
    img[FN] = [255, 165, 0]     # laranja

    # Mostrar imagem
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("TP: branco | TN: preto | FN: laranja | FP: vermelho")
    plt.axis("off")
    plt.show()



def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()




def measure_inference_speed(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)

    # Warm-up (opcional, mas recomendado com CUDA)
    warmup_inputs, _ = next(iter(test_loader))
    warmup_inputs = warmup_inputs.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(warmup_inputs)

    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            torch.cuda.synchronize()
            start_time = time.time()

            _ = model(inputs)

            torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)
            total_samples += batch_size

    avg_time_per_image = total_time / total_samples
    time_per_image = f"{avg_time_per_image * 1000:.3f} ms"
    fps = f"{1.0 / avg_time_per_image:.2f}"
    return fps, time_per_image
