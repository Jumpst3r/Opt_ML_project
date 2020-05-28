# some helper functions for the plot
import torch
import matplotlib.pyplot as plt

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_results_pso(swarm, RESULT_PATH, img_id, show_reduce=False):
    # plt.rcParams["axes.titlesize"] = 8
    plt.tight_layout()
    plt.axis('off')
    if show_reduce: plt.subplot(1, 3, 1)
    else: plt.subplot(1, 2, 1)
    plt.title("Prediction: " + CIFAR_CLASSES[swarm.TRUECLASS.item()], fontsize=18)
    img = (swarm.target_image[0].detach().permute(1, 2, 0).cpu()).detach()
    fig1 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)))
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    if show_reduce:
        plt.subplot(1, 3, 2)
        plt.title(f"Prediction: {CIFAR_CLASSES[swarm.predicted_label.item()]} \n(before reduction L2={swarm.get_l2(swarm.before_reduce):.2f})", fontsize=18)
        img = swarm.before_reduce.view(
            swarm.channelNb, swarm.width, swarm.height).permute(1, 2, 0).cpu().detach()
        fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))
        
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
    if show_reduce: plt.subplot(1, 3, 3)
    else: plt.subplot(1, 2, 2)
   
    plt.title("Prediction: " + CIFAR_CLASSES[swarm.predicted_label.item()] + (f"\n(after reduction L2={swarm.get_l2():.2f})" if show_reduce else (f"\n(L2={swarm.get_l2():.2f})")), fontsize=18)
    img = swarm.best_particle_position.view(
        swarm.channelNb, swarm.width, swarm.height).permute(1, 2, 0).cpu().detach()
    fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))

    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    # Save the image to the results/ folder
    plt.savefig(RESULT_PATH + f"result_pso_{img_id}.png")
    plt.clf()

def plot_results_torchattacks(target_im, pre, TRUECLASS, output, l2, attack_str, RESULT_PATH, img_id):
    # plt.rcParams["axes.titlesize"] = 50
    plt.tight_layout()
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.title("Prediction: " + CIFAR_CLASSES[TRUECLASS.item()], fontsize=18)
    img = (target_im[0].detach().permute(1, 2, 0).cpu()).detach()
    fig1 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)))
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    
    plt.subplot(1, 2, 2)
   
    plt.title("Prediction: " + CIFAR_CLASSES[pre.item()] + f"\n(L2={l2:.2f})", fontsize=18)
    img = output.view(
        target_im.shape[1], target_im.shape[2], target_im.shape[3]).permute(1, 2, 0).cpu().detach()
    fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))

    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    # Save the image to the results/ folder
    plt.savefig(RESULT_PATH + f"result_{attack_str}_{img_id}.png")
    plt.clf()