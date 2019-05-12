"""
This toolkit help you save and load
variable that would save you tons of time
"""
import pickle


def save_obj(obj, name):
    """
    This function is used to save an object
    :type obj: any type of object would work
    :type name: a string which you'd like to name your saved file to
    """
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    This function is used to load a saved object
    :type name: a string that specify the name of your saved file. E.g: /obj/name.pkl
    """

    with open('obj/' + name + '.pkl', 'rb') as f: return pickle.load(f)
