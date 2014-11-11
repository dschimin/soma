#general imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#method imports
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from astroML.classification import GMMBayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from astroML.utils import completeness_contamination

#load data
#mus,omegaes,omegae1s,omegals,alphas,rcircs,masses2,ls,tsnaps
data = np.load('./ch9_data.npy')
classifications = np.load('./ch9_classifications.npy')
omegae1s = data[:,2]
omegals = data[:,3]

#plot all data with classifications
if 1:
	scatter_color=True
	plt.figure(figsize=(6,6), dpi=120)
	plt.subplot(aspect='equal')
	isstream = (np.array(classifications) =='stream')
	isshell = (np.array(classifications) =='shell')
	istooearly = (np.array(classifications) =='tooearly')
	istoolate = (np.array(classifications) =='toolate')
	isunsure = (np.array(classifications) =='unsure')
	classes=[isstream, isshell, isunsure]
	if scatter_color:
	    colors =['b', 'r', 'g']
	    markers = ['^', 'o', 's']
	    labels = ['stream', 'shell', 'unsure']
	    edgecolors=['none', 'none', 'none']
	else:
	    colors =['k', 'w', '0.5']
	    markers = ['^', 'o', 's']
	    labels = ['stream', 'shell', 'unsure']
	    edgecolors=['none', 'k', 'none']
	for i in np.arange(len(classes)):
	    plt.scatter(np.array(omegae1s)[classes[i]]*180./np.pi, np.array(omegals)[classes[i]]*180./np.pi, 
	    	label = labels[i], marker = markers[i], color = colors[i], edgecolor=edgecolors[i],s=25)
	plt.plot([0,100],[0,100],color='k',lw=3)
	plt.xlim([0,100])
	plt.ylim([0,100])
	plt.xlabel('$\Psi_E^1$ [degrees]', size = 'x-large')
	plt.ylabel('$\Psi_L$ [degrees]', size = 'x-large')
	plt.legend(['$\mathrm{\mu}=1$']+labels, scatterpoints=1)

#pick training set
trainers = np.random.choice(np.arange(199),199,replace=False)

#choose model
model = GaussianNB()
#model = LDA()
#model = QDA()
#model = GMMBayes(3)
#model = KNeighborsClassifier(8)
#model = LogisticRegression(class_weight='auto')
#############model = LinearSVC()

def do_stuff(model):
	#train
	model.fit(np.transpose([omegals[trainers], omegae1s[trainers]]), classifications[trainers])

	#predict all classifications
	predictions = model.predict(np.transpose([omegals, omegae1s]))

	# predict the classification probabilities on a grid
	xlim = (0,100*np.pi/180.)
	ylim = (0,100*np.pi/180.)
	xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1],1000),
	                     np.linspace(ylim[0], ylim[1],1000))
	Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

	#color grid by prediction
	gridclass = np.zeros(Z[:,1].shape)
	gridclass[(Z[:,0]>Z[:,1])&(Z[:,0]>Z[:,2])]=0
	gridclass[(Z[:,1]>Z[:,2])&(Z[:,1]>Z[:,0])]=1
	gridclass[(Z[:,2]>Z[:,0])&(Z[:,2]>Z[:,1])]=2
	gridclass=gridclass.reshape(xx.shape)

	clf()
	cmap = matplotlib.colors.ListedColormap(['red', 'blue', 'green'])
	bounds=[0,.666,1.333,2]
	norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
	img = plt.imshow(np.transpose(gridclass), origin='lower',interpolation='nearest', 
		cmap = cmap, extent=[0,100,0,100], alpha = 0.5)
	cbar = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[.25,1,1.75])
	cbar.ax.set_yticklabels(['shell', 'stream', 'unsure'])# horizontal colorbar

	#overplot correct answers
	plt.scatter(omegae1s[classifications=='stream']*180./np.pi, omegals[classifications=='stream']*180./np.pi, color='blue' , s=75, edgecolor='w')
	plt.scatter(omegae1s[classifications=='shell']*180./np.pi,  omegals[classifications=='shell']*180./np.pi,   color='red' , s=75, edgecolor='w')
	plt.scatter(omegae1s[classifications=='unsure']*180./np.pi, omegals[classifications=='unsure']*180./np.pi, color='green', s=75, edgecolor='w')
	plt.xlim([0,100])
	plt.ylim([0,100])
	plt.xlabel('$\Psi_E^1$ [degrees]', size = 'x-large')
	plt.ylabel('$\Psi_L$ [degrees]', size = 'x-large')

	print model
	for name in ['stream', 'shell ', 'unsure']:
		com, con = completeness_contamination(predictions==name, classifications==name)
		print name + ' completeness:%1.2f; contamination:%1.2f'%(com, con)
