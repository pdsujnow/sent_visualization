import pandas
import yaml
import cPickle

from os import path

def load_from_file(category, fname, sent_label='sentence', cat_label='label'):
    """Load data from file

    Parameters
    ----------
    category : the name of category
    fname : input file name

    Returns
    -------
    A pandas.DataFrame containing sentences listed in the file, all labeled as category.

    """
    with open(fname) as f:
        try:
            res = cPickle.load(f)
        except:
            res = pandas.DataFrame({cat_label: category, sent_label: f.readlines()})
    return res


def load(directory, spec_fname='export.yaml'):
    assert(path.isdir(directory)), '{} is not a directory'.format(directory)
    with open(path.join(directory, spec_fname)) as f:
        spec = yaml.load(f)

    frames = []

    for category, dest in spec.items():
        dest = path.join(directory, dest)
        if path.isfile(dest):
            frames.append(load_from_file(category, dest))
        elif path.isdir(dest):
            print dest
        else:
            assert False, "{} not Found".format(dest)
    return pandas.concat(frames)
