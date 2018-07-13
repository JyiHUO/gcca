from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as ds

digital = ds.load_boston()

train = digital.data
target = digital.target

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)


cca = CCA(n_components=5)
cca.fit(X_train, y_train)

X_c, Y_c = cca.transform(X_train, y_train)