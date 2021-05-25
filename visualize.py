import matplotlib.pyplot as plt

def plot(img, boxes):
    fig, ax = plt.subplots(1, dpi=96)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    width, height, _ = img.shape

    ax.imshow(img, cmap='gray')
    fig.set_size_inches(width / 80, height / 80)

    for box in boxes:
        rect = plt.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        fill=False,
        linewidth=1.0)
        ax.add_patch(rect)
 
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from dataset import train_dataset    
    img, target = train_dataset[1].values()
    plot(img, target.get("boxes").tolist() if target.get("boxes") is not None else [])