from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# tsne plot for below word
# for_word = 'food'
def tsne_plot(for_word, w2v_model):
    # trained fastText model dimention
    dim_size = w2v_model.wv.vectors.shape[1]

    arrays = np.empty((0, dim_size), dtype='f')
    word_labels = [for_word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, w2v_model.wv.__getitem__([for_word]), axis=0)

    # gets list of most similar words
    sim_words = w2v_model.wv.most_similar(for_word, topn=10)

    # adds the vector for each of the closest words to the array
    for wrd_score in sim_words:
        wrd_vector = w2v_model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # ---------------------- Apply PCA and tsne to reduce dimention --------------

    # fit 2d PCA model to the similar word vectors
    model_pca = PCA(n_components=10).fit_transform(arrays)

    # Finds 2d coordinates t-SNE
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(model_pca)

    # Sets everything up to plot
    df_plot = pd.DataFrame({'x': [x for x in Y[:, 0]],
                            'y': [y for y in Y[:, 1]],
                            'words_name': word_labels,
                            'words_color': color_list})

    # ------------------------- tsne plot Python -----------------------------------

    # plot dots with color and position
    plot_dot = sns.regplot(data=df_plot,
                           x="x",
                           y="y",
                           fit_reg=False,
                           marker="o",
                           scatter_kws={'s': 40,
                                        'facecolors': df_plot['words_color']
                                        }
                           )

    # Adds annotations with color one by one with a loop
    for line in range(0, df_plot.shape[0]):
        plot_dot.text(df_plot["x"][line],
                      df_plot['y'][line],
                      '  ' + df_plot["words_name"][line].title(),
                      horizontalalignment='left',
                      verticalalignment='bottom', size='medium',
                      color=df_plot['words_color'][line],
                      weight='normal'
                      ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for word "{}'.format(for_word.title()) + '"')
    plt.show()


fast_Text_model = Word2Vec.load("ft_model_yelp")

print(fast_Text_model.wv['τσιρκο'])

print(fast_Text_model.wv.most_similar('ξεφτιλες', topn=10))

tsne_plot(for_word='ξεφτιλες', w2v_model=fast_Text_model)