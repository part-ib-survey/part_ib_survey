import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def dataframe_to_numpy(data):
    N, Col = data.shape

    B_choices = np.full((N, 19), -1)
    A_choices = np.full((N, 4), -1)

    for row in data.iterrows():
        i, series = row

        # omit first and last element
        # these are the time stamp and part IA choices
        for subject, score in enumerate(series[1:-1]):
            B_choices[i, subject] = responses_B[score]

        A_choices_list = series[-1].split(', ')
        for j, subject in enumerate(A_choices_list):
            A_choices[i, j] = responses_A[subject]

    return B_choices, A_choices

def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels, rotation='vertical')

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center")

    if save:
        plt.savefig(save + 'StackedPlot.png')
    plt.show()

def produce_stacked_plot():
    clean_B_choices = B_choices.astype(np.int32)
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=5), 0, clean_B_choices)[::-1]
    order = counts[:-1].sum(axis=0).argsort()[::-1]
    counts = counts.T[order].T

    series_labels = ['5 - Certain', 4, 3, 2, '1 - Impossible']
    categories = list(B_abbreviations.keys())
    categories = [categories[key] for key in order]

    plot_stacked_bar(counts, series_labels, category_labels=categories, grid=False)

def B_choices_to_probability(choices):
    probs = choices**2 / (np.sum(choices**2, axis=1)+1e-8)[:, np.newaxis]
    probs = np.sqrt(probs)*np.sqrt(3)
    return probs

def A_choices_to_probability(choices):
    N = len(choices)
    S = choices.max()+1

    out = np.full((N, S), 0)
    for i in range(N):
        out[i][choices[i]] = 1

    return out

def probs_to_heatmap(sqrt_probs_X, sqrt_probs_Y, diag=False):
    NX, SX = sqrt_probs_X.shape
    NY, SY = sqrt_probs_Y.shape
    assert NX == NY, 'Must have same number of samples'
    probs_X = sqrt_probs_X**2
    probs_Y = sqrt_probs_Y**2

    numerator = probs_Y.T.dot(probs_X)
    denominator = np.sum(probs_X, axis=0)
    out = numerator/(denominator+1e-8)
    if diag == True:
        out[np.arange(SX), np.arange(SY)] = 1
    return out

def heatmap_from_probs(probs_X, probs_Y, categories_X, categories_Y, diag=False, square=False, file_name='heatmap.png'):
    heatmap = probs_to_heatmap(probs_X, probs_Y, diag=diag)
    ax = sns.heatmap(heatmap, square=square)
    ind_X = np.arange(heatmap.shape[1])+0.5
    ind_Y = np.arange(heatmap.shape[0])+0.5
    ax.set_xticks(ind_X)
    ax.set_yticks(ind_Y)
    ax.set_xticklabels(categories_X, rotation='vertical')
    ax.set_yticklabels(categories_Y, rotation='horizontal')
    ax.invert_xaxis()
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_title('P(Y|X)')
    if save:
        plt.savefig(save + file_name)
    plt.show()


def get_descriptions_B(choices):
    B_subjects_number = np.array(list(B_abbreviations.keys()))
    choice_strings = []
    for person in choices:
        subjects = person.nonzero()[0]

        subject_strings = B_subjects_number[subjects]
        subject_scores = (person[person!=0]+1).astype(str)

        person_string = "\n".join(map(': '.join, zip(subject_strings, subject_scores)))
        choice_strings.append(person_string)
    return choice_strings

def plot_pca_B(file_name='pca.html'):
    pca = PCA(n_components=2)
    pca.fit(B_probs)
    reduced_dims = pca.transform(B_probs)
    texts = get_descriptions_B(B_choices)
    data = go.Scatter(x=reduced_dims[:,0], y=reduced_dims[:,1], mode='markers',text=texts)
    fig = go.Figure(data=data)
    fig.update_traces(marker=dict(size=18,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.write_html(save + file_name)
    fig.show()


# if you are reading this, i apologize because this is the worst code i've written
sns.set()

# set to None for no saving
save = "Figures/"

responses_B = {
    "1 - Impossible": 0,
    2: 1,
    3: 2,
    4: 3,
    "5 - Certain": 4,
}
responses_A = {
    'Biology of Cells': 0,
    'Chemistry': 1,
    'Computer Science': 2,
    'Earth Sciences': 3,
    'Evolution and Behaviour': 4,
    'Materials Science': 5,
    'Physics': 6,
    'Physiology of Organisms': 7,
    'Mathematics A': 8,
    'Mathematics B': 9,
    'Mathematical Biology': 10,
}
B_abbreviations = OrderedDict([
    ('BMB', 'Biochemistry and Molecular Biology'),
    ('BoD', 'Biology of Disease'),
    ('CDB', 'Cell and Developmental Biology'),
    ('Chem A', 'Chemistry A'),
    ('Chem B', 'Chemistry B'),
    ('Earth A', 'Earth Sciences A'),
    ('Earth B', 'Earth Sciences B'),
    ('Ecology', 'Ecology, Evolution & Conservation (formerly Ecology)'),
    ('Animal Bio', 'Evolution & Animal Diversity (formerly Animal Biology)'),
    ('Psych', 'Experimental Psychology'),
    ('HPS', 'History and Philosophy of Science'),
    ('Materials', 'Materials Science'),
    ('Maths', 'Mathematics'),
    ('Neuro', 'Neurobiology'),
    ('Pharma', 'Pharmacology'),
    ('Phys A', 'Physics A'),
    ('Phys B', 'Physics B'),
    ('Physiology', 'Physiology'),
    ('Plants', 'Plants and Microbial Sciences'),
])
A_abbreviations = OrderedDict([
    ('BoC', 'Biology of Cells'),
    ('Chem', 'Chemistry'),
    ('Compsci', 'Computer Science'),
    ('Earth', 'Earth Sciences'),
    ('E&B', 'Evolution and Behaviour'),
    ('Materials', 'Materials Science'),
    ('Physics', 'Physics'),
    ('PoO', 'Physiology of Organisms'),
    ('Maths A', 'Mathematics A'),
    ('Maths B', 'Mathematics B'),
    ('Math Bio', 'Mathematical Biology'),
])

data = pd.read_excel(r'AnonData.xlsx')
B_choices, A_choices = dataframe_to_numpy(data)
B_probs = B_choices_to_probability(B_choices)
A_probs = A_choices_to_probability(A_choices)
categories_B = list(B_abbreviations.keys())
categories_A = list(A_abbreviations.keys())

plot_pca_B()
produce_stacked_plot()
heatmap_from_probs(B_probs, B_probs, categories_B, categories_B, square=True, diag=True, file_name='heatmap_B')
heatmap_from_probs(A_probs, A_probs, categories_A, categories_A, square=True, diag=True, file_name='heatmap_A')
heatmap_from_probs(A_probs, B_probs, categories_A, categories_B, file_name='heatmap_AB')