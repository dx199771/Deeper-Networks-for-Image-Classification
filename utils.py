"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""
import matplotlib.pyplot as plt

def plot_loss(validation_loss,train_loss,name,display=False):
    """
    Plot one model's loss on one figure
    :param validation_loss: validation loss value
    :param train_loss: training loss value
    :param name: model name
    :param display: whether display or not
    :return: None
    """
    # create new figure
    f2, axs2 = plt.subplots(figsize=(12, 12))
    axs2.set_title('Training and validation losses on {}'.format(name))
    axs2.set_xlabel('Steps')
    axs2.set_ylabel('Loss')

    # plot training loss and validation loss
    axs2.plot(train_loss, '-r', label='Training loss')
    axs2.plot(validation_loss, '-b', label='Validation loss')
    axs2.legend(loc="upper right")
    # y axis value limitation
    axs2.set_ylim([0,4])
    axs2.grid()
    f2.savefig('./results/{}_loss.png'.format(name))
    if display:
        plt.show()

def plot_error(error,name,display=False):
    """
    Plot one model's error on one figure
    :param error: validation error value
    :param name: validation model name
    :param display: whether display or not
    :return: None
    """
    # create new figure
    f3, axs3 = plt.subplots(figsize=(12, 12))
    axs3.set_title('Validation error rate on {}'.format(name))
    axs3.set_xlabel('Steps')
    axs3.set_ylabel('Error %')

    axs3.plot(error, '-r', label='Validation error rate')
    axs3.legend(loc="upper right")
    axs3.grid()
    f3.savefig('./results/{}_error.png'.format(name))
    if display:
        plt.show()


def plot_all_errors(testing_models,colour, validation_error):
    """
    Plot all validation errors on one figure
    :param testing_models: testing model name
    :param colour:  colour for plotting
    :param validation_error: validation error value
    :return: None
    """
    plt.figure("Training error rate")
    plt.title('Training error rate')
    plt.xlabel('Steps')
    plt.ylabel('Error %')

    plt.plot(validation_error, colour, label=testing_models)
    plt.legend(loc="upper right") # set label
    plt.savefig('./results/error.png')

def plot_all_losses(testing_models, colour, trining_loss):
    """
    Plot all training and validation losses on one figure
    :param testing_models: testing model name
    :param colour: colour for plotting
    :param trining_loss: loss value of training
    :return: None
    """
    plt.figure("Training loss")
    plt.title('Training loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    # y axis value limitation
    plt.ylim([0,4])
    plt.plot(trining_loss, colour, label=testing_models)
    plt.legend(loc="upper right") # set label
    plt.savefig('./results/loss.png')



def predicted_image(input_img,input_label,output_label,dataset):
    """
    Plot predicted image in one figure
    :param input_img: input image set
    :param input_label: input image labels
    :param output_label: predicted image labels
    :param dataset: MNIST or CIFAR-10 dataset name
    :return: None
    """
    # create figure
    f, axs = plt.subplots(5, 5,figsize=(12, 12))
    axs = axs.flatten()
    for index in range(len(axs)):
        # set up label colour
        if input_label[index].item() == output_label[index].item():
            label_color = {'family':'serif','color':'red','size':12}
        else:
            label_color = {'family': 'serif', 'color': 'black', 'size': 12}
        if dataset == "CIFAR10":
            label_to_name = {0:"airplane",
                             1:"automobile",
                             2:"bird",
                             3:"cat",
                             4:"deer",
                             5:"dog",
                             6:"frog",
                             7:"horse",
                             8:"ship",
                             9:"truck"}
            # set title name
            axs[index].set_title("Label:{}\nPredict:{}".format(label_to_name[input_label[index].item()],
                                                               label_to_name[output_label[index].item()]),
                                 fontdict=label_color)
        else:
            axs[index].set_title("Label:{}\nPredict:{}".format(input_label[index].item(),
                                                               output_label[index].item()),
                             fontdict=label_color)
        input_img_ = input_img[index].permute(1, 2, 0)
        # denormalize
        axs[index].imshow(input_img_* 0.5 + 0.5)
    f.tight_layout()
    f.savefig('./results/predicted.png')
    #plt.show()