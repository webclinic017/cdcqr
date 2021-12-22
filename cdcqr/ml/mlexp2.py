import sys
sys.path.append('/core/github/cryptoderiv-quant/')
import vectorbt as vbt

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from ct.fs import Classifier1ifLarger


modelsdict={
    "rf":{
        'model':RandomForestClassifier(**{'min_samples_split': 2, 'min_samples_leaf': 0.01, 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 4, 'max_features': 0.7, 'max_depth': 4, 'ccp_alpha': 0.0}),
        'paramsd':
            {
            'n_estimators':[1,5,10,100],
            'max_depth': [2,3,4,10],
            'bootstrap':[True,False],
            'min_samples_split':[2,0.01,0.1,0.3,0.5,0.6],
            'min_samples_leaf':[1],#,0.01,0.05,0.1],
            'ccp_alpha':[0.0,0.01,0.1,0.2,0.5],
            'max_features':[0.5,0.7,1.0],
            'max_leaf_nodes':[2,4,5,10,20],
            'min_impurity_decrease':[0.0,0.01,0.1,0.2],
            'class_weight':["balanced","balanced_subsample",{1:1},{1:.1},{1:1000}]#,{1:100},{1:1000},{1:10000},{1:0.1}]
            },
        'fixedparams':{},    
    },
    "dt":{
        'model':DecisionTreeClassifier(),
        'paramsd':{
            'max_depth': [2,3],#,4,10],
            'min_samples_split':[2],#,0.01,0.1,0.3,0.5,0.6],
            'min_samples_leaf':[1],#,0.01,0.05,0.1],
            'ccp_alpha':[0.0],#,0.01,0.1,0.2,0.5],
            'max_features':[0.5,0.7,1.0],
            'max_leaf_nodes':[2,4,5,10,20],
            'min_impurity_decrease':[0.0],#,0.01,0.1,0.2],
            'class_weight':[{1:1},{1:.1},{1:1000}]#,{1:100},{1:1000},{1:10000},{1:0.1}]
            },
        
        'fixedparams':{},
    },
    "oneiflarger":{
        'model':Classifier1ifLarger(thresh=1,quantile=0.99),
        'paramsd':{'quantile':[0.8,0.9,0.99,0.999,0.9999],'thresh':[0.3,0.5,0.9,1],'how':['all','any']},
        'fixedparams':{},
        'params':{}
    },
    "bernoullinb":{
        'model':BernoulliNB(),
        'fixedparams':{},
        'paramsd':{'alpha':[0,1.0,0.1,1e-02,1e-3]}
    },
    
    "gaussiannb":
    {
        'model':GaussianNB(),
        'paramsd':{'var_smoothing':[1e-09,1e-08]},
        'fixedparams':{},
    },
    "svc":
    {
        'model':SVC(gamma='auto'),
        'paramsd':{'C':[1.0,0.1,1e-02,1e-3,0.0]}
    },
    
    "kmeans":
    {
        'model':KNeighborsClassifier(),
        'fixedparams':{},
        'paramsd':{'n_neighbors':[2,3,4,5,10,20],'leaf_size':[30,10,100]},
    },
    "dummy":
    {
            'model':DummyClassifier(),
            'fixedparams':{},
            'paramsd':{'strategy':["stratified","most_frequent", "prior", "uniform", "constant"]}
    },
        
        
    "mylog":
    {
        'model':MyLogisticRegression(tol=0.0001, C=1.0, solver='saga', max_iter=300), #solver='lbfgs' 
        'fixedparams':dict(penalty='l2',l1_ratio=None,class_weight='balanced',verbose=0),
        'paramsd':{
                'tol':[0.0001,0.001,0.01],
                'C': [ 0.1, 1,10],
                'max_iter': [10,50,100,300],
                'solver':['lbfgs', 'sag','saga']
                }
        
    },
    "mlp":
    {
        'model':MLPClassifier(alpha=1, max_iter=100),
        'paramsd':{'alpha':[0.1,1],'max_iter':[10,100,200]},
        'fixedparams':{}
    }
    
}



from scipy.stats import hmean
def mlexp(sym='ETH',qsuffix='USDT@binancealt',date=[date(2021,1,2),date(2021,10,15)],ytrain_thresh=6,ytrain_periods=100,ytest_thresh=3,ytest_periods=100,vdelay=2,ewmlags=[10,100,1000],ytypes=["ys"],xtypes=["v","q3","l"],qlags=[3000],request=[],models=["oneiflarger","dt"],randcv_n_iter=50,sl=3,tp=3,ts=None,vbtfreq=None):
    """
    q3 quantiles3
    l  log(1+x) tranform (for positive vals like inflow)
    
    request:
         plottree
         backtrader: run backtrader get sharpe for different trailpercents,tp,sl
         
    """
    logging.error(f"mlexp start:{sym} {date}  {ytypes} {xtypes} {models}")
    res=[]
    
    sym=sym                 
    try:
        dfgsi=glassnode('metrics~transactions~transfers_volume_to_exchanges_mean',a=sym,i='10m',s=date[0].isoformat(),price=False).asfreq("10Min")
        df0=qbars(sym+qsuffix,freq="10Min",table='ppttrades',date=date,rdb=False).asfreq("10Min") #addwhere=', vol>0.01',
    except Exception as e:
        logging.error(f"glassnode or qbars error:sym={sym} qsuffix={qsuffix} {str(e)}")
        return []
    y=df0.addysbarrier(thresh=ytrain_thresh,periods=ytrain_periods,inplace=True,retfnames=True)[0]
    yt=df0.addysbarrier(thresh=ytest_thresh,periods=ytest_periods,inplace=True,retfnames=True)[0]

    if 'sr' in ytypes:
        df['sr+1']=np.sign(df['rl+1.c'])
        df['sr+10']=np.sign(df['rl+10.c'])
        df['sr+100']=np.sign(df['rl+100.c'])
        ys.extend(['sr+1','sr+10','sr+100'])
    
    df=df0.drop(columns=['v']).join(dfgsi)#.adddiffs(cols=['c'],method='ys',lags=[-1,-5,-10,-100,-1000],dropna=False,).adddiffs(cols=['c'],method='rl',lags=[-1,-5,-10,-100,-1000],dropna=False,)
    df['vd']=df['v'].shift(vdelay)
    
    if 'corr' in request:
        res1=[]
        for v in np.linspace(1,1000,100):
            r=df.query('vd>@v').corr(method='spearman')['vd']
            re1s.append({'vd':v,'rho+1':r.loc['rl+1.c'],'rho+5':r.loc['rl+5.c'],'rho+10':r.loc['rl+10.c'],'rho+100':r.loc['rl+100.c'],'rho+1000':r.loc['rl+1000.c']})
        DF(res1).set_index('vd').plot()
        
        res1=[]
        for v in np.linspace(1,1000,100):
            r=df.query('vd>@v')[['ys+1.c','ys+10.c','ys+100.c']].mean()
            res1.append({'vd':v,'rho+1':r.loc['ys+1.c'],'rho+10':r.loc['ys+10.c'],'rho+100':r.loc['ys+100.c']})
        DF(res1).set_index('vd').plot()
    
    allxs=['vd']
    
    if 'q99' in xtypes:
        xq99=df.addqs([0.0,0.99,1.0],lags=qlags,cols=['vd'],inplace=True,retfnames=True,fillnans=-1)
        allxs.extend(xq99)
        
    if 'q5' in xtypes:
        xq5=df.addqs(5,lags=qlags,cols=['vd'],inplace=True,retfnames=True,fillnans=0)
        allxs.extend(xq5)
    
    if 'rsi' in xtypes:
        pass
        #add rsi, other momentum
        
    if 'volatility' in xtypes:
        pass
        #addvol

    #df['qrandn0.0,0.999,1.0,3000.v2']+=1
    #set below median to 0, to clear noise 
    
    vdmed=df.vd.median()
    df['vdm']=df['vd']
    df.loc[df.vd<df.vd.median(),'vdm']=0
    
    if 'l' in xtypes:
        df['l.vd']=np.log(1+df['vd'])
        allxs.append('l.vd')
    
    if 'ewm' in xtypes:
        xsewm=df.addma(cols=['l.vd'] if 'l' in xtypes else ['vd'],lags=ewmlags,method='ewm',inplace=True,retfnames=True)
        allxs.extend(xsewm)

    for x in allxs:
        
        xs=[x]
        dftrain=df.iloc[:int(0.7*len(df))].dropna()
        dftest=df.iloc[-int(0.2*len(df)):].dropna()
        try:
            cv=TimeSeriesSplit(gap=100, max_train_size=len(dftrain)//2,n_splits=3, test_size=len(dftrain)//4) #1min 30 secs
        except Exception as e:
            print("run cv again with smaller number of splits")
            raise(e)

        if 'showsplit' in request:
            showsplit(dftrain,cv,y=xs[0]) #y=y
            showsplit(dftrain,cv,y=y) #y=y

        if df[y].value_counts(normalize=True).prod()<0.2:
            print('consider using classweight')

        if 'hist' in request:
            df[y].value_counts()
            df[y].value_counts(normalize=True)
            df[xs].hist()
            df[xs].value_counts()
            df[xs].value_counts(normalize=True)

        if 'eda' in request:
            dfeda(df.dropna(),y=ytrain,xs=['vd'])
            dfeda(df.dropna(),y=ytest,xs=['vd'])

        if 'plot' in request:
            dftest[[y]].plot(figsize=(22,5),alpha=0.5)
            dftest[[y]+xs].scale().plot(figsize=(22,5),alpha=0.5)
            dftrain[[y]+xs].scale().plot(figsize=(22,5),alpha=0.5)

        #sample_weight=np.where(dftrain[xs]==1,100,1).flatten()
        #sample_weight

        bestdummyp=0

        for modelstr in ['dummy']+models: #rf etc
            modeld=modelsdict[modelstr]
            print(modeld)
            model=modeld['model']
            model.set_params(**modeld['fixedparams'])
            fit_params={}#'sample_weight':sample_weight}
            try:
                randcv = RandomizedSearchCV(model, param_distributions=modeld['paramsd'], n_iter=randcv_n_iter, scoring='precision', n_jobs=10, cv=cv, verbose=0, random_state=1,refit=True).fit(dftrain[xs], dftrain[y],**fit_params)
            except Exception as e:
                res.append({'sym':sym,'model':modelstr,'xs':xs,'y':y,'yt':yt,'err':'randcv:'+str(e)})
                continue
                
            #print(f"rscv {model.__class__.__name__} fixedparams={fixedparams} bestscore={randcv.best_score_} bestparams={randcv.best_params_} ")
            display(DF(randcv.cv_results_).sort_values(by='mean_test_score',ascending=False)[['mean_test_score' ,'std_test_score', 'params','split0_test_score','split1_test_score','split2_test_score']].head(3))
            #print(f"rscv \n")
            #display(DF(randcv.cv_results_).sort_values(by='mean_test_score',ascending=False))
            #xgbrf=xgb.XGBRFClassifier(**fixedparams,**randcv.best_params_)
            #model=DecisionTreeClassifier(**fixedparams,**randcv.best_params_)
            model.set_params(**randcv.best_params_)
            meancvscore=randcv.best_score_
            print(model)
            #ipdb.set_trace()
            try:
                resscor=runscoring(df=dftrain,y=y,yt=yt,xs=xs,model=model,nansy='.fillna(0)',nansx=None,scoring='precision',verbose=100,cv=cv,dftest=dftest,request=request,sl=sl,tp=tp,ts=ts,vbtfreq=vbtfreq) #'cm' 'plottree'
            except Exception as e:
                res.append({'sym':sym,'model':modelstr,'xs':xs,'y':y,'yt':yt,'bestcvscore':meancvscore,'err':str(e)})
                logging.error({'sym':sym,'model':modelstr,'xs':xs,'y':y,'yt':yt,'bestcvscore':meancvscore,'err':str(e)})
                continue
                
            
            print('cv:',resscor['cv'])
            print(f"test p{resscor['p']},recall {resscor['recall']}")
            print(f"test pis{resscor['pis']},recallis {resscor['recallis']}")
            if 'tree' in resscor:print(resscor['tree']) 
            if 'viz' in resscor:display(resscor['viz'])
            
            hm=hmean([resscor['p'],resscor['pis'],meancvscore])
              
            resd={}
            resd['hm']=hm
            
            if modelstr=='dummy' and bestdummyp<hm:
                bestdummyhm=hm
                
            if modelstr!='dummy':
                resd['hm_margin']=hm-bestdummyhm
            
            #log tree
            if resscor['p']>0.01:
                
                if 'vbt' in request and resscor['vbtntpm']<0.5:# and resscor['recall']<1.0:
                    continue #do not log infrequent model
                if 'coef' in resscor:resd['coef']=resscor['coef']
                
                if 'vbt' in request:
                    
                    resd['rpa']=resscor['rpa']
                    resd['vbtp']=resscor['vbtp']
                    resd['vbtsr']=resscor['vbtsr']
                    resd['vbtntpm']=resscor['vbtntpm']
                    resd['vbtinfo']=resscor['vbtinfo']
                    
                res.append({**{'sym':sym,'model':modelstr,'xs':str(xs),'y':y,'yt':yt,'p':resscor['p'],'pyt':resscor['pyt'],'pis':resscor['pis'],'ymean':resscor['ymean'],'ymeanis':resscor['ymeanis'],'ntpm':resscor['ntpm'],'ntpmis':resscor['ntpmis'],'recall':resscor['recall'],'recallis':resscor['recallis'],'bestcvscore':np.round(meancvscore,2),'bestparams':randcv.best_params_},**resd})    
                logging.error({**{'sym':sym,'model':modelstr,'xs':str(xs),'y':y,'yt':yt,'p':resscor['p'],'pyt':resscor['pyt'],'pis':resscor['pis'],'ymean':resscor['ymean'],'ymeanis':resscor['ymeanis'],'ntpm':resscor['ntpm'],'ntpmis':resscor['ntpmis'],'recall':resscor['recall'],'recallis':resscor['recallis'],'bestcvscore':np.round(meancvscore,2),'bestparams':randcv.best_params_},**resd})    
                
    return res

