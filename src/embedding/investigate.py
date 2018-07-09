from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tsne_plot(word_emb_dict, filename=None):
    """Creates a TSNE model and plots it
    insert
        # sen = [' '.join(t) for t in self.X]
        # sen_dict = {s: e for s, e in zip(sen, self.X_tfidf)}
        # sen_dict[' '.join(X[0])] = target_tfidf[0]
        # tsne_plot(sen_dict, '/plots/{}.pdf'.format('_'.join(X[0])))
    into the predit function of embedding to plot query question in
    relation to training set
    """
    labels = []
    tokens = []

    for word, emb in word_emb_dict.items():
        tokens.append(emb)
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    f = plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i] ,y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    fout = filename if filename else "tsne_embedding.pdf"
    f.savefig(fout, bbox_inches='tight')
