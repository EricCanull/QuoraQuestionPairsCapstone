import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def pie_chart(train_df):
        
    # Pie chart duplicate question distribution
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    data = np.sum(train_df.iloc[:, -1] == 1)
    labels = 'Duplicates', 'Non-duplicates'
    sizes = [data, train_df.shape[0] - data]
    pie_colors = "#9688f6", "#117A65"

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(sizes, 
                                      explode=(0, 0.1), 
                                      labels=labels,
                                      colors=pie_colors, 
                                      rotatelabels=False,
                                      autopct=lambda pct: func(pct, data),
                                      shadow=True, 
                                      startangle=90
                                      )

    ax.legend(wedges, labels,
              title="Questions",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.title("Duplicate vs Non-Duplicate Questions", bbox={'facecolor':'0.8', 'pad':5})
    plt.setp(autotexts, size=12, weight="bold")
    plt.setp(texts, size=12, weight="bold")
    plt.show()


def histogram(title, q1, q2):
    plt.figure(figsize=(12, 5))
    plt.hist(q1, bins=100, normed = True, color='green', label='Q1')
    plt.hist(q2, bins=100, normed = True, color='blue',alpha = 0.5, label='Q2')
    plt.title(title, fontsize=15)
    plt.legend()
    plt.xlabel(title, fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.tight_layout()
    plt.show()


def heatmap_chart(df, features):    
    sns.heatmap(df[features].corr(), annot=True,
                cmap=sns.color_palette("cubehelix", 8),
                linewidths=0.1, cbar_kws={'label': 'Correlations', 
                                            'orientation': 'horizontal'})
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.show()


def accuracy_chart(epochs, acc, val_acc, loss, val_loss):
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
