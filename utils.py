import matplotlib.pyplot as plt

def plot_loss(loss,display=False):
    plt.figure(0)

    plt.title('Training loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.plot(loss, 'g', label='Training loss')
    if display:
        plt.show()
    plt.savefig('loss.png')

def plot_error(error,display=False):
    plt.figure(1)

    plt.title('Training loss')
    plt.xlabel('Steps')
    plt.ylabel('Error %')

    plt.plot(error, 'g', label='Training error')
    if display:
        plt.show()
    plt.savefig('error.png')

