import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim.corpora import Dictionary
import joblib
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

SOME_FREE_SONGTEXT = "So one night I said to Kathy We gotta get away somehow Go somewhere south and somewhere warm But for God's sake let's go now. And Kathy she sort of looks at me And asks where I wanna go So I look back and I hear me say I don't care but we gotta go chorus and key change And all the other people Who slepwalk thru their days Just sort of faded out of sight When we two drove away And ev'ry day we travelled We were lookin' to get wise And we learned what was the truth And we learned what were the lies And in LA we bought a bus Sort of old and not too smart So for six hundred and fifty bucks We got out and made a start We hit the road down to the South And drove into Mexico That old bus was some old wreck But it just kept us on the road. chorus etc We drove up to Alabam And a farmer gave us some jobs We worked them crops all night and day And at night we slept like dogs We got paid and Kathy said to me It's time to make a move again And when I looked into her eyes I saw more than a friend. chorus etc And now we've stopped our travels And we sold the bus in Texas And we made our home in Austin And for sure it ain't no palace And Kathy and me we settled down And now our first kid's on the way Kathy and me and that old bus We did real good to get away."

