import matplotlib.pyplot as plt

def plot_loss(loss):
    plt.figure(0)

    plt.title('Training loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.plot(loss, 'g', label='Training loss')
    #plt.show()
    plt.savefig('loss.png')

def plot_error(error):
    plt.figure(1)

    plt.title('Training loss')
    plt.xlabel('Steps')
    plt.ylabel('Error %')

    plt.plot(error, 'g', label='Training error')
    #plt.show()
    plt.savefig('error.png')