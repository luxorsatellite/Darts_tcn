# Forecasting Models, Dr. Loukas Samaras, 2025, University of Alcalá de Henares
# Temporal Convolution Network (TCN) vs. Linear Regression Model (LR), 2025. Train single series with covariates (past data)
ver='1.03'; print(ver) # #support for darts 0.29, 0.32 and new darts 0.36, MacOs, Ubuntu 24 amd Windows, seasonality graph
#https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
#darts 0.32: if layers >0: weight_norm=False (Replaced the deprecated torch.nn.utils.weight_norm function with torch.nn.utils.parametrizations.weight_norm)
import torch; print('torch version:', torch.__version__); torch.set_float32_matmul_precision('highest')
import darts; darts_version=darts.__version__;print('darts version:', darts_version) #  0.29.0: to_dataframe'. Did you mean: 'pd_dataframe'?
darts_ver=darts_version[:4]
import numpy as np; print('numpy version:', np.__version__)
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from darts.models import TCNModel, LinearRegressionModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mape, rmse
import timeit
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging #logging.disable(logging.CRITICAL) #disable showing model params, size end errors

class globals:
  def __init__(self):
    model_backtest_retrain_time=''

#data
filename='enstsoe_solar_weather.csv'
target_col=0
input_cols=[0,3,4,8] #[0,3,4,8], [0,3,4,8,-2,-1], []:None
resamplimg_interval='h' #'h', '15Min'
index_col=0
drop_null=0 #0:no, 1:yes
part_size=1 #is head
delimiter=','
country='ES'

test_ratio=0.20 #0.2
test_size=0 #0: all in test ratio, >0: fixed (e.g 240)
backtest_size=0 #backtest: all in test size, >0: fixed (e.g 240)
plot_len=10*24 #values to plot, 0:all test size
seasonality_check=0 #0:no, 1:yes
seasonality_order=24*7 #check seasonality (max lags) if seasonality_check
go_back=0 #0:no (default), 1: forward, -1 backward
convert_fp32=1  #0:no (default), 32:yes (convert to float32)
model_val_set_adjust=0 #0:no, 1: yes (dilation base >1 or/and more epochs)

evaluate_network=0 #0:no (evaluate only LR model), 1:yes
backtest_retrain=0 #0:no, 1:yes, production mode: manually retrain all previous values with best epoch. Produces batches of forecasts out of the sample of size=forecast horizon 
retrain_lr_tuner=1 #0:no, 1:yes, find best lr for every retrain step
future_network=1 #0:no, 1:yes (predict future out of the sample with TCN)
future_values=24 # production mode: values out of the sample
remove_negatives=1 #0:no, 1:yes, negative values are not accepted in production node

ticker='energy'
leverage=1 #levarage ratio to estimate profits of Stock Market transactions

#settings
seed=0 #0, 42
cpu_device=0 #0:cuda, 1:cpu
model_name='TCN'
INPUT_LEN=24+1 # seasonal periods: 0: all test size-OUTPUT_CHUNK_LENGTH=24*7, 24+1: out=1
lags=24**2 #lags for linear regression (24**2, 24*7)
NR_EPOCHS_VAL_PERIOD = 1 #**optional
FORECAST_HORIZON=1  # < INPUT_LEN (>=future values)
OUTPUT_CHUNK_LENGTH=FORECAST_HORIZON # < INPUT_LEN
BATCH_SIZE=24; num_layers=4; kernel_size=24; num_filters=24; weight_norm=True; dilation_base=1; lr=1e-3; lr_decay=0  \
    ;dropout=0.2; MAX_SAMPLES_PER_TS=None; torch_optim='NAdam' # Adam (default),NAdam,SGD,Adadelta,AdamW,LBFGS,Adagrad,Adamax,ASGD,RAdam,RMSprop,(SparseAdam,Optimizer,Rprop)
learning_rate_tuner=1 #0:no 1:yes 0.0010964781961431851 for test=0.2 (run seperately if darts ver> 0.29 (2.0.0< torch version <2.2.0)) #may require more epochs
#from torch> 2.0.0, https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
#optimizers: Adam (default),NAdam,SGD,Adadelta,AdamW,LBFGS,Adagrad,Adamax,ASGD,RAdam,RMSprop,(SparseAdam,Optimizer,Rprop)

retrain=0 #retrain steps **********
if retrain > 0:
    logging.disable(logging.CRITICAL) #disable showing model params, size end errors

#early stopper
MAX_N_EPOCHS = 100 #100, 30
MIN_DELTA=0.000001 #0.000001, 0.0000001
PATIENCE=20 #20
stop_loss='val_loss' if retrain==0 else 'train_loss'

scaler=Scaler() #scaler = Scaler(scaler=MinMaxScaler()) #Scaler(scaler=MaxAbsScaler() or MinMaxScaler
#encoders={'cyclic': {'past': ['month','week','day','dayofweek','hour']},'transformer': scaler} #None, if no encoders
encoders={'cyclic': {'past': ['month','week','dayofyear','day','dayofweek','hour']},'transformer': scaler}
encoders={'cyclic': {'past': ['dayofweek']},'transformer': scaler} if part_size < 0.2 else encoders # for smaller size

#set random seeds
import random
def set_seed(seed=seed) -> None:  
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #or CUBLAS_WORKSPACE_CONFIG=:16:8 # with dfilation base and layers > 1 #deterministic
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) ##    
    random.seed(seed)
    np.random.seed(seed)
set_seed()

#start timer
start_all = timeit.default_timer()

#1.Data Preparation---------------------------------------
df = pd.read_csv(filename,index_col=index_col, delimiter=delimiter).select_dtypes(['number'])
df=df.dropna() if drop_null==1 else df
df.index.name='datetime'
df.index = pd.to_datetime(df.index)    
df.index = df.index - pd.Timedelta(hours=go_back) if go_back!=0 else df.index

#show duplicate indices, resample, interpolate and sort
if len(df[df.index.duplicated()])>0:
    print('duplicate indices:'); print(len(df[df.index.duplicated()])) 
    #print(df[df.index.duplicated()]) #all duplicate rows     
    df = df[~df.index.duplicated(keep='last')] #remove duplicate indices
    print('duplicates removed')

if not df.index.is_monotonic_increasing:
    df = df.sort_index()
    print('index sorted')

df=df.resample(resamplimg_interval).interpolate(limit_direction='both')

if part_size < 1 and future_values > 2: #hold backup
    df_all_data=df.copy()
df=df.head(int(len(df)*part_size))

#all data
print('all data:')
print(df)
print('---------------------')

#find holidays and weekends
import holidays
hldy = holidays.country_holidays(country)
df['isHoliday'] = df.index.to_series().apply(lambda x: x in hldy) 
df['isHoliday'] = df['isHoliday'].replace({True: 1, False: 0}).astype(int) #binary
df['is_weekend'] = (df.index.day_of_week >= 5).astype(int) #binary #weekend

print('data with datetime attributes:')
#print(df)
print('-------------------------------')

#select data
print('selected data:')
print(df.iloc[:,[target_col]])
print('---------------------')
series_cols=df.columns
'''
#select dates
df=df[(df.index >= '2018-05-01') & (df.index < '2018-10-08')]
'''
if convert_fp32 > 0:
    print('converting to fp32...')
    df=df.astype('float32')    
    # or for series: all_series = [s.astype(np.float32) for s in all_series]

#set series and covariates
all_series = darts.TimeSeries.from_dataframe(df.iloc[:,[target_col]]) #, fill_missing_dates=True, freq=resamplimg_interval)

# Split in train/val/test according to test ratio
test_size=int(len(df)*test_ratio) if test_size==0 else test_size
train, val = all_series[:-test_size],  all_series[-test_size:]

#lengths
print('data length:', len(df), 'train.len:', len(train), 'val.len:', len(val))

#plot series
#data frames
df_train=train.to_dataframe() if int(darts_ver[2:4])> 32 else train.pd_dataframe()
df_val=val.to_dataframe() if int(darts_ver[2:4])> 32 else val.pd_dataframe()
df_series=pd.concat([df_train,df_val],axis=1)

plt.figure(figsize=(15, 8))
plt.grid()
df_series.iloc[:,0].plot(label="Training Data", color='b')
df_series.iloc[:,1].plot(label="Validation Data", color='r')
plt.legend()
plt.title('Time-series: ' + df.columns[target_col])
plt.xlabel('')
plt.savefig('Plot1_0_Time-series')
plt.close()

#descriptives
print('----------------------------------')
print('decrptive statistics:')
print(df.iloc[:,[target_col]].describe())
print('----------------------------------')
df.iloc[:,[target_col]].describe().to_excel('descriptives.xlsx', index=True)

#get target col name and substitute invalid chrs   
target_col_name=df.columns[target_col]
import re
target_col_name=res = re.sub(r'[^a-zA-Z0-9]', '', target_col_name)

#find correlations
import statistics
from scipy import stats
print('----------------------------------')

cors=[]
cols=[]
cor_count=0
correlation_line='correlation of '+ df.columns[target_col]+ ':'
print(correlation_line)
for col in df.columns:
    try:
        cor_count+=1
        cor=stats.pearsonr(df[df.columns[target_col]], df[col]) #scipy stats
        cors.append(cor)
        cols.append(col)
    except Exception as error:
        cor_count-=1

print('----------------------------------')  
df_cors=pd.DataFrame(cors,columns=['statistic', 'pvalue'])
df_cols=pd.DataFrame(cols, columns=['var'])
df_correlations=pd.concat([df_cols,df_cors], axis=1)
df_correlations.index.name='n'
print(df_correlations)
df_correlations.to_excel('correlations_'+target_col_name+'.xlsx')

#sorted by abs
df_correlations_abs_sorted=df_correlations
df_correlations_abs_sorted['statistic']=abs(df_correlations_abs_sorted['statistic'])
df_correlations_abs_sorted=df_correlations_abs_sorted.sort_values(by='statistic', ascending=False)
df_correlations_abs_sorted.to_excel('correlations_sorted_'+target_col_name+'.xlsx')

#correlation matrix heat map
correlation_matrix = df[df.columns].corr()
import seaborn as sns
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.title('correlation heatmap')
plt.savefig('plot1_correlation_matrix')
plt.close()

if seasonality_check==1: # Part 2. Check seasonality    
    print('checking seasonality...')
    seasonal_orders=[]
    found=''
    start_season_time = timeit.default_timer()
    for m in range(2, seasonality_order): #for all data: len(df)
        seasonal, period = darts.utils.statistics.check_seasonality(train, m=m,
                                                                    max_lag=seasonality_order,
                                                                    alpha=0.05)
        if seasonal:
            end_season_time = timeit.default_timer()
            season_time=round(end_season_time-start_season_time, 2)
            print("Seasonality of order:", str(period), season_time, 'sec')
            seasonal_orders.append(period)
            found='yes'

    if found=='':    
        print('no seasonal order found. Order 1 will be used')
        period=1
        seasonal_orders.append(1)

    df_orders=pd.DataFrame(seasonal_orders, columns=['order'])
    df_orders.index.name='check'
    df_orders.to_csv('df_seasonal_orders.csv')
    print(df_orders)

    #plot seasonality
    x=list(range(len(df_orders))); y=df_orders[df_orders.columns[0]].tolist()
    def add_labels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha='center')  # Aligning text at center

    plt.plot(df_orders, label='seasonality', c='b')
    add_labels(x,y)
    plt.legend(loc='upper right')
    plt.title('Seasonality')
    plt.xlabel('order')
    plt.ylabel('time ('+resamplimg_interval+')')
    plt.savefig('Plot1_2_seasonality')
    plt.close()

    print('Seasonality point (last):', period)
    #k = seasonality_order #seasonality order (period, default:361)
    print('order point used:', seasonal_orders[-1])
    k = period #seasonality order (period, default:361)
    k = seasonal_orders[-1] #seasonality order (period, default:361)
    print('-------------')

    #check current seasonality of train data, #max_lag must be greater than or equal to 1 and less than len(ts).
    from darts.utils.statistics import check_seasonality

    # find all seasonalities for train data for every season interval
    sesonalities=[]
    print('find all season frequencies:')
    def seasonalities(df, interval, test_ratio=test_ratio):
        current_saeasonality=''
        df=df.resample(interval).interpolate(limit_direction='both')
        interval_name=interval_names[interval]
        series = darts.TimeSeries.from_dataframe(df)
        
        train, val = series.split_before(1-test_ratio)
        try:
            current_saeasonality=check_seasonality(train, max_lag=240)
        
        except Exception as e:
            print(e)
            current_saeasonality=['n/a', 0]
            interval_name=interval_name
        
        print('check_seasonality for ', interval_name +':', 'length: (all:' +str(len(df)) + ', train:' + str(len(train)) + ')', 
                current_saeasonality) 

        del series; del val   
        return current_saeasonality, interval_name, len(train)
        
    intervals=['y','m','w','d','h'] #dayofweek 'W-Mon'
    interval_names={'y': 'year','m': 'month','w': 'week', 'd': 'day', 'h': 'hour'}

    intervals_names=[]
    train_lengths=[]
    for i in range (len(intervals)):
        seasonality, interval_name, train_length = seasonalities(df.iloc[:,[target_col]], intervals[i])
        sesonalities.append(seasonality)    
        intervals_names.append(interval_name)
        train_lengths.append(train_length)

    df_sesonalities=pd.DataFrame(sesonalities, columns=['True','freq'])
    df_interval_names=pd.DataFrame(intervals_names, columns=['interval'])
    df_lengths=pd.DataFrame(train_lengths, columns=['length'])
            
    df_sesonalities=pd.concat([df_interval_names, df_sesonalities, df_lengths], axis=1)
    df_sesonalities.index = np.arange(1, len(df_sesonalities)+1) #reset index from 1
    df_sesonalities.index.name='#'

    df_sesonalities.to_excel('seasonalities_'+target_col_name+'.xlsx')
    print('----------------------------------------')

print('scaling ...')
if len(input_cols)>0: #if covariates exist, scale first covariates to preserve scaler transorfmation
    past_covariates=darts.TimeSeries.from_dataframe(df.iloc[:,input_cols]) #,fill_missing_dates=True, freq=resamplimg_interval)
    cov_train, cov_val = past_covariates[:-test_size], past_covariates[-test_size:]    
    print('-----------')
    print('covariates:')
    print(past_covariates.to_dataframe()) if int(darts_ver[2:4])> 32 else print(past_covariates.pd_dataframe())
    past_covariates = scaler.fit_transform(past_covariates) ##scale covariates
    cov_train=past_covariates[:-len(val)]
    cov_val = past_covariates[-len(val):]
    covariates_past_constructor=[-lags]   
else:
    covariates_past_constructor=None
    past_covariates = None
    cov_train=None
    cov_val = None
    covariates_past_constructor=None

# Scale after covariates to preserve transformation
train = scaler.fit_transform(train)
val = scaler.transform(val)
all_series=scaler.transform(all_series)

#define start date if all_series is used
backtest_size=len(val) if backtest_size==0 else backtest_size
print('backtest_size:', backtest_size)
start_date_point=len(df)-backtest_size-FORECAST_HORIZON+1

start_date=df.iloc[start_date_point:start_date_point+1].index.item()
#print('start_date:', start_date, pd.Timestamp(start_date))

plot_len=len(val) if plot_len==0 else plot_len

def eval_model(name, is_retrain, val_set=val):
    print('evaluating...')
    backtest_start=timeit.default_timer()
    
    if is_retrain==0:    
        preds = model.historical_forecasts(
            series=all_series, #val without start date or all_series with start_date and overlap=false
            past_covariates=past_covariates,
            start=pd.Timestamp(start_date),
            forecast_horizon=FORECAST_HORIZON,
            retrain=retrain,
            stride=1,
            verbose=True,
            #overlap_end=True
        )
        
        #time elapsed
        backtest_time=timeit.default_timer()-backtest_start
        backtest_time = '{:0,.2f}'.format(backtest_time)        
        
        #inverse scaler
        val_set=scaler.inverse_transform(val_set)        
        preds=scaler.inverse_transform(preds)
        
    else:
        preds=darts.TimeSeries.from_dataframe(df_retrain)
        val_set=scaler.inverse_transform(val_set) #inverse only validation set        
        backtest_time=globals.model_backtest_retrain_time        
      
    #dataframes
    df_val=val_set.to_dataframe() if int(darts_ver[2:4])> 32 else val_set.pd_dataframe()
    df_preds=preds.to_dataframe() if int(darts_ver[2:4])> 32 else preds.pd_dataframe()
    df_result=pd.concat([df_val.tail(len(preds)), df_preds], axis=1)        
    df_result.columns=['actual','forecast']
    df_result.index.name='datetime'
    df_result.to_excel('df_result_'+name+'.xlsx')
       
    #symmetric MAPE
    smapes = smape(preds, val_set)
    sMAPE='{:.2f}%'.format(np.mean(smapes))
    
    #mape
    MAPE_darts='{:.2f}%'.format(mape(preds, val_set))
    
    #rmse
    rmse_darts='{:.2f}'.format(rmse(preds, val_set))
       
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    mape_score=100*mean_absolute_percentage_error(df_result.iloc[:,0],df_result.iloc[:,1])
    mape_score = '{:0,.2f}%'.format(mape_score)

    #normalized mae
    import statistics
    meanX=statistics.mean(df_result.iloc[:,0])
    mae_stats=mean_absolute_error(df_result.iloc[:,0],df_result.iloc[:,1])    
    mae=round(100*mae_stats/meanX,2) #Mean Absolute Error of the Mean)
    mae='{:.2f}%'.format(mae) #mae score
    mae_stats="{:.2f}".format(mae_stats)
    
    #Root Mean Square Error (RMSE)
    import math
    rmse_score='{:0,.2f}'.format(math.sqrt(mean_squared_error(df_result.iloc[:,0],df_result.iloc[:,1])))
        
    #correlation
    from scipy import stats
    r2_score_value_scipy=stats.pearsonr(df_result.iloc[:,0].tail(len(preds)),df_result.iloc[:,1].tail(len(preds))) #scipy stats
  
    #calculate success of prediction (directional succes)
    #find starting starting col: if success_start_col= target then the previous value
    df_result['start('+df_result.columns[0]+')']=df_result.iloc[:,[0]].shift() # column 3 (start). [1]].shift(): previous forecast, [0]: previous actual
    df_result['real_diff']=df_result.iloc[:,0]-df_result.iloc[:,2] #column 4 (03): (real-starting column)    
    df_result['exp_diff']=df_result.iloc[:,1]-df_result.iloc[:,2] # column 5 (04): expected-starting col        
    df_result['real_diff_sign']=np.sign(df_result['real_diff'])
    df_result['exp_diff_sign']=np.sign(df_result['exp_diff'])
    df_result['success']=0

    ix=0; count=0
    for _ in df_result.iterrows():        
        if df_result.loc[df_result.index[ix], 'real_diff'] !=0 and \
            df_result.loc[df_result.index[ix], 'exp_diff'] !=0: #if not zero (if change exists.)
            count+=1 #only if change >0 or <0
            
            if df_result.loc[df_result.index[ix], 'exp_diff_sign']==df_result.loc[df_result.index[ix], 'real_diff_sign']:
                df_result.loc[df_result.index[ix],'success']=1
        ix+=1

    total_success=df_result['success'].sum()
    total_success_per=round(100*total_success/(count-1),2)
    total_success_per = '{:0,.2f}%'.format(total_success_per)
    
    #profits
    df_result['profit']=0
    ix=0
    for _ in df_result.iterrows():        
        if df_result.loc[df_result.index[ix], 'real_diff'] !=0:
            real_diff=abs(df_result.loc[df_result.index[ix], 'real_diff']) #absolute value
            start_col_name=df_result.columns[2]        
            start_value=df_result.loc[df_result.index[ix], start_col_name]
            success=df_result.loc[df_result.index[ix], 'success']
            profit=real_diff/start_value            
            
            if success==0:
                df_result.loc[df_result.index[ix], 'profit']=-profit
            else:
                df_result.loc[df_result.index[ix], 'profit']=profit
        else:
            df_result.loc[df_result.index[ix], 'profit']=0
        ix+=1

    df_result['profit']=df_result['profit']*leverage
    df_result['profits']=df_result['profit'].cumsum()
    total_profit='{:0,.2f}'.format(df_result['profit'].sum())

    title= 'v.'+ver+ ', '+ ticker+' '+ name+ ', backtest series: '+ df.columns[target_col] + ', length:' + str(len(df)) +', backtest length:' +\
        str(len(preds))+ ', forecast horizon:' + str(FORECAST_HORIZON) +  '\n' + \
            'sMAPE: '+ sMAPE + ', nMAE: ' + mae + ', MAPE: '+ MAPE_darts +', correlation: '+str(r2_score_value_scipy[0]) + \
            ', success: '+ total_success_per+ ', profits: ' + total_profit + '\n' +\
                'training time: '+ model_training_time + ' sec' + ', backtest_time: ' + backtest_time + ' sec'

    print(title) #results    
    
    #plot profits
    plt.figure(figsize=(15, 8)) #21,16
    plt.grid()  
    df_result.iloc[:,-1].tail(plot_len).plot(label='profits',c='b')
    plt.legend(loc='upper right')
    plt.title(df.columns[target_col]+ ' profits:'+ total_profit)
    plt.xlabel('')          
    plt.savefig('plot2b_profits_'+name)
    plt.close()      
    
    #plot resulta
    plt.figure(figsize=(15, 8)) #21,16
    plt.grid() 
    df_result.iloc[:,0].tail(plot_len).plot(label='actual',c='b')
    df_result.iloc[:,1].tail(plot_len).plot(label='forecast',c='r')        
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('')           
    plt.savefig('plot2_'+name)
    plt.close()    
   
    #add result metrics to file
    INPUT_LAGS= lags if name=='linear regression' else INPUT_LEN
    
    lines_to_write=[
            'version                           '+str(ver),
            '----------------------------------',
            'target:                           '+ df.columns[target_col],
            'input variables:                  '+ ', '.join(df.iloc[:,input_cols].columns),
            'input length:                     '+str(INPUT_LAGS),
            'total size:                       '+str(len(df)),
            'mean (X):                         '+'{:0,.2f}'.format(df.iloc[:,[target_col]].mean().item()) +', standard deviation (X): '+'{:0,.2f}'.format(df.iloc[:,[target_col]].std().item()),
            'train size:                       '+str(len(train)),
            'test size:                        '+str(len(val)),
            'model:                            '+str(model),
            'number of samples:                '+str(MAX_SAMPLES_PER_TS),
            'train time:                       '+str(model_training_time) + ' sec',
            '----------------------------------',
            'forecast horizon:                 '+str(FORECAST_HORIZON),
            'backtest size:                    '+str(len(df_result)),
            'rmse (Darts):                     '+str(rmse_darts),
            'rmse:                             '+str(rmse_score),
            'sMAPE:                            '+sMAPE,
            'mape_score (Darts):               '+MAPE_darts,
            'mape_score:                       '+str(mape_score),
            'MAE:                              '+str(mae_stats),
            'nMAE:                             '+str(mae),  
            'correlation (r2/pvalue):          '+str(r2_score_value_scipy[0])+ ' / '+str(str(r2_score_value_scipy[1])),
            'success (%):                      '+total_success_per,
            'backtest_time:                    '+str(backtest_time)+' sec',
            '----------------------------------'
            ]
        
    #write results to file
    with open('statistics_'+name+'.txt', 'w') as f:
        for line in lines_to_write:
            #print(line)
            f.write(line)
            f.write('\n')   

#Build a Simple Linear Model
model = LinearRegressionModel(lags=lags, 
                                 output_chunk_length=OUTPUT_CHUNK_LENGTH, 
                                 output_chunk_shift=0,
                                 lags_past_covariates=covariates_past_constructor, #postive integer or a list
                                 lags_future_covariates=None,
                                 add_encoders=encoders,
                                 )

start_model=timeit.default_timer()
print('--------------------------------')
print('linear regression:')
print('--------------------------------')
model.fit(train, past_covariates=past_covariates, future_covariates=None, max_samples_per_ts=None)
model_training_time="{:.2f}".format(timeit.default_timer()-start_model)
eval_model("linear regression",0)
print('--------------------------------')
print('end of linear regression backtest')
print('--------------------------------')

INPUT_LEN=len(val)-OUTPUT_CHUNK_LENGTH if INPUT_LEN==0 else INPUT_LEN

def build_fit_tcn_model(): #Build TCN Model 

    # reproducibility    
    set_seed() #torch.manual_seed(seed)  
    start_model = timeit.default_timer()

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping(stop_loss, min_delta=MIN_DELTA, patience=PATIENCE, verbose=True)
    callbacks = [early_stopper]
    
    if cpu_device==0:    
        pl_trainer_kwargs = {"callbacks": callbacks}
    else:
        pl_trainer_kwargs = {"callbacks": callbacks, "accelerator": "cpu"}
    
    #optimizer
    optimizer_kwargs={'lr': lr, 'weight_decay': 0}
    
    if torch_optim=='Adam':
        optimizer_cls=torch.optim.Adam #1
    elif torch_optim=='SGD':
        optimizer_cls=torch.optim.SGD #2
    elif torch_optim=='Adadelta':
        optimizer_cls=torch.optim.Adadelta #3
    elif torch_optim=='AdamW':
        optimizer_cls=torch.optim.AdamW #4
    elif torch_optim=='Adagrad':
        optimizer_cls=torch.optim.Adagrad #7
    elif torch_optim=='Adamax':
        optimizer_cls=torch.optim.Adamax #8
    elif torch_optim=='ASGD':
        optimizer_cls=torch.optim.ASGD #9
    elif torch_optim=='NAdam':
        optimizer_cls=torch.optim.NAdam #10
    elif torch_optim=='RAdam':
        optimizer_cls=torch.optim.RAdam #11
    elif torch_optim=='RMSprop':
        optimizer_cls=torch.optim.RMSprop #12
    else:
        optimizer_cls=torch.optim.Optimizer #13

    #sanity check
    model = TCNModel(
        model_name=model_name,
        random_state=seed,
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        batch_size=BATCH_SIZE,
        optimizer_cls=optimizer_cls,
        add_encoders=encoders,
        )
    
    print('sanity_check_predict_likelihood_parameters...') 
    parameters=model._sanity_check_predict_likelihood_parameters
    print(str(parameters)[84:-1])     

    with open('sanity_check.txt', 'w') as f:
        f.write(str(parameters)[84:-1])
        f.write('\n')

    if learning_rate_tuner==1:
        # build the TCN model to find the best learning rate (24 parameters)
        model = TCNModel(
            model_name=model_name,
            random_state=seed,
            input_chunk_length=INPUT_LEN,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            output_chunk_shift=0,
            batch_size=BATCH_SIZE,
            n_epochs=MAX_N_EPOCHS,
            nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
            num_filters=num_filters,
            num_layers=num_layers,
            weight_norm=weight_norm,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            optimizer_kwargs={'lr': lr, 'weight_decay': lr_decay}, #lr_decay: 0 (default)
            optimizer_cls=optimizer_cls,
            add_encoders=encoders,
            #likelihood=None,
            pl_trainer_kwargs=pl_trainer_kwargs,
            log_tensorboard=True,
            force_reset=True,
            save_checkpoints=True,
            lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau, #optional
            lr_scheduler_kwargs={'threshold': 0.0,  'verbose' : True, 'monitor': stop_loss}, #'train_loss' if not validation or 'val_loss'
        )

        # run the learning rate tuner
        # may increase the number of epochs if the tuner did not give satisfactory results
        print('---------------------')
        scaler_tuner=Scaler() #MinMax scaler
        if retrain_lr_tuner==0: #use train size
            train_lr = scaler_tuner.fit_transform(train)
            val_lr = scaler_tuner.transform(val)
        
        else: #use all size
            train_lr = scaler_tuner.fit_transform(train_nn_data)
            val_lr = scaler_tuner.transform(val_nn_data)
        
        results = model.lr_find(series=train_lr, val_series=val_lr)    
        print('learning rate tuner results:', results)
        print('---------------------')
        
        optimizer_kwargs={'lr': results.suggestion(), 'weight_decay': lr_decay}
        lr_final=results.suggestion()
        print(lr_final)
        
        # plot the results
        plt.rcParams["figure.figsize"] = (8,5)
        results.plot(suggest=True, show=False)
        plt.title('learning rate tuner: '+ str(lr_final))
        plt.legend()
        plt.savefig('Plot0c_lr_tuner')
        plt.close()        
       
    else:
        optimizer_kwargs={'lr': lr, 'weight_decay': lr_decay}
        lr_final=lr
    
    #build final model
    model = TCNModel( 
        model_name=model_name,       
        random_state=seed, #optional
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        output_chunk_shift=0,
        batch_size=BATCH_SIZE, #optional **
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD, #optional
        num_filters=num_filters, #optional
        num_layers=num_layers, #optional
        weight_norm=weight_norm, #optional **
        kernel_size=kernel_size, #optional **
        dilation_base=dilation_base, #optional **
        dropout=dropout, #optional **
        optimizer_kwargs=optimizer_kwargs, #optional
        optimizer_cls=optimizer_cls, #optional
        add_encoders=encoders, #optional
        #likelihood=None, #optional
        pl_trainer_kwargs=pl_trainer_kwargs, #optional       
        log_tensorboard=True, #optional
        force_reset=True, #optional
        save_checkpoints=True,
        lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau, #optional
        lr_scheduler_kwargs={'threshold': 0.0,  'verbose' : True, 'monitor': stop_loss}, #'train_loss' if not validation or 'val_loss, #optional
    )

    print('model used:')
    print(model)
    print('------------------------------------------------------------')   
  
    # train the model
    model.fit(
        series=train_nn_data,
        val_series=val_nn_data,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
        past_covariates=past_covariates,
        val_past_covariates=past_covariates,
        #num_loader_workers=num_workers, #use only if computer is very slow or occupied with another proccess
    )

    model_training_time="{:.2f}".format(timeit.default_timer()-start_model)

    # reload best model over course of training
    model = TCNModel.load_from_checkpoint(model_name)
    return model, model_training_time

#set model train and val
train_nn_data=train

if model_val_set_adjust==1:
    #define val set (scale series)
    # when validating during training, we can use a slightly longer validation set which also contains the first input_chunk_length time steps
    val_nn_data = all_series[-((2 * test_size) + INPUT_LEN) : -test_size]
else:
    val_nn_data=val

if evaluate_network==1:
    print('----------------------------------')
    print('model_val_set:', len(val_nn_data))
    print('----------------------------------')

    model, model_training_time = build_fit_tcn_model()
    eval_model('TCN_model',0)

'''
---------------------------------------------------------
Part 2. backtest retrain with best epoch 
manual retrain of all previous data, as in production mode
2a. Linear Model
---------------------------------------------------------
'''
if backtest_retrain==1:
    all_series_backup=all_series
    OUTPUT_CHUNK_LENGTH=FORECAST_HORIZON
    print('--------------------')

    #1. Linear Regression Model
    print('Linear Regression retrain:')
    df_retrain=pd.DataFrame()
    start_backtest_retrain = timeit.default_timer()
    retrain_step=1
    model = LinearRegressionModel(lags=lags, 
                                 output_chunk_length=OUTPUT_CHUNK_LENGTH, 
                                 output_chunk_shift=0,
                                 lags_past_covariates=covariates_past_constructor, #postive integer or a list
                                 lags_future_covariates=None,
                                 add_encoders=encoders)
    
    for i in range(len(val),0,-FORECAST_HORIZON): #start, end, step
        retraining_time_step=timeit.default_timer()
        all_series=all_series_backup[:-i]
        print('retrain step:',retrain_step, 'end_date:', (all_series.to_dataframe().index.max())) \
            if int(darts_ver[2:4])> 32 else print('retrain step:',retrain_step, 'end_date:', (all_series.pd_dataframe().index.max()))

        #train and fit all previous values
        train_nn_data=all_series
        val_nn_data=all_series
        
        #Linear Regression Model
        model.fit(train_nn_data, past_covariates=past_covariates, future_covariates=None, max_samples_per_ts=None)
        prediction_retrain_values = model.predict(n=FORECAST_HORIZON, num_samples=1, past_covariates=past_covariates)
        model_training_time="{:.2f}".format(timeit.default_timer()-start_model)
        
        prediction_retrain_values=scaler.inverse_transform(prediction_retrain_values)
        df_retrain=pd.concat([df_retrain,prediction_retrain_values.to_dataframe()]) \
            if int(darts_ver[2:4])> 32 else pd.concat([df_retrain,prediction_retrain_values.pd_dataframe()])
        #print(df_retrain)
        
        model_training_time=timeit.default_timer()-retraining_time_step
        model_training_time='{:.2f}'.format(model_training_time)
        #print('--------------------')
        retrain_step+=1
    
    #results    
    df_retrain.columns=['forecast']
    df_retrain=df_retrain.head(len(val))
    print('retrain forecast:')    
    print(df_retrain)
    print('--------------------------')
    
    globals.model_backtest_retrain_time = '{:.2f}'.format(timeit.default_timer()-start_backtest_retrain)
    print('Linear Regression retraining time:', globals.model_backtest_retrain_time,'sec')
    eval_model('LR_model_retrain',1) #1 prediction retrain
    
    #2. TCN Model
    if evaluate_network==1:
        print('----------------------------')
        print('TCN model retrain')    
        #logging.disable(logging.CRITICAL) #disable showing model params and size 
        retrain_step=1
        df_retrain=pd.DataFrame()
        start_backtest_retrain = timeit.default_timer()
        
        for i in range(len(val),0,-FORECAST_HORIZON):
            #set_seed()
            retraining_time_step=timeit.default_timer()
            
            #train and val all previous data
            all_series=all_series_backup[:-i]
            train_nn_data=all_series 
            val_nn_data=all_series        
            
            print('retrain step:',retrain_step, 'end_date:', (all_series.to_dataframe().index.max())) \
                if int(darts_ver[2:4])> 32 else print('retrain step:',retrain_step, 'end_date:', (all_series.pd_dataframe().index.max()))
                
            model, model_training_time = build_fit_tcn_model()

            print('--------------------')
            print('backtest re-training time:', model_training_time, 'sec')
            print('--------------------')
                
            #produce predictions
            prediction_retrain_values = model.predict(n=FORECAST_HORIZON, num_samples=1, past_covariates=past_covariates)
            prediction_retrain_values=scaler.inverse_transform(prediction_retrain_values)
            df_retrain=pd.concat([df_retrain,prediction_retrain_values.to_dataframe()]) \
                if int(darts_ver[2:4])> 32 else pd.concat([df_retrain,prediction_retrain_values.pd_dataframe()])
            
            retrain_step+=1
        
        df_retrain.columns=['forecast']
        df_retrain=df_retrain.head(len(val))
        print('retrain forecast:')    
        print(df_retrain)
        print('--------------------------')
        
        globals.model_backtest_retrain_time = '{:.2f}'.format(timeit.default_timer()-start_backtest_retrain)
        print('retraining time:', globals.model_backtest_retrain_time,'sec')
        eval_model('TCN_model_retrain',1) #1 prediction retrain

'''
----------------------------------
PART 3. Forecast out of the sample
----------------------------------
'''
#predict next values out of the sample by fitting again, but now the entire series
if future_values > 0:
    print('--------------------------------------------')  
    print('5. finding future values (out of the sample)')
    print('--------------------------------------------')
    #logging.disable(logging.CRITICAL) #disable showing model params and size
    INPUT_LEN=kernel_size**2 if future_values > FORECAST_HORIZON else INPUT_LEN #increase input length for larger OUTPUT_CHUNK_LENGTH
    OUTPUT_CHUNK_LENGTH=future_values
    model_name=model_name+'_future'    
    
    #train and fit all values
    train_nn_data=all_series_backup if backtest_retrain==1 else all_series #val without start date or all_series with start_date and overlap=true  
    val_nn_data=all_series_backup if backtest_retrain==1 else all_series
    
    print('model_train_val:', len(val_nn_data))
    print('----------------------------------')
    
    start_future = timeit.default_timer()
    
    if future_network==1: #use neural network
        model, model_training_time = build_fit_tcn_model()    
        prediction = model.predict(n=future_values,num_samples=1, past_covariates=past_covariates)
        future_model_name='TCN'
        
    else: #use Linear Regression Model
        logging.disable(logging.CRITICAL) #disable showing model params, size end errors
        model = LinearRegressionModel(lags=lags, 
                                      output_chunk_length=OUTPUT_CHUNK_LENGTH, 
                                      output_chunk_shift=0,
                                      lags_past_covariates=[-lags], #postive integer or a list
                                      lags_future_covariates=None,
                                      add_encoders=encoders)       
        
        model.fit(train_nn_data, past_covariates=past_covariates, future_covariates=None, max_samples_per_ts=None)
        prediction = model.predict(n=future_values, num_samples=1, past_covariates=past_covariates)
        model_training_time="{:.2f}".format(timeit.default_timer()-start_model)
        future_model_name='LR'
    
    print('future training time:', model_training_time, 'sec')
    print('--------------------')
    
    try:
        #unscale and create df
        prediction=scaler.inverse_transform(prediction)
        df_future_prediction=prediction.to_dataframe() if int(darts_ver[2:4])> 32 else prediction.pd_dataframe()

        #replaces the negative numbers with zeros
        df_future_prediction[df_future_prediction < 0] = 0 if remove_negatives==1 else df_future_prediction

        df_future_prediction.to_excel('df_future_prediction.xlsx')
        #add last value to continue plot
        df_future_prediction_backup=df_future_prediction.copy() #hold backup before adding      
                
        df_future_prediction=pd.concat([df.iloc[:,[target_col]].tail(1), df_future_prediction])        
        
        #concat history and future
        df_future_prediction=pd.concat([df.iloc[:,[target_col]].tail(plot_len),df_future_prediction], axis=1)
        df_future_prediction.columns=[df.columns[target_col],'forecast (future)']        
        
        plt.figure(figsize=(15, 8))
        plt.grid()
        df_future_prediction.iloc[:,0].plot(label='actual',c='b')
        df_future_prediction.iloc[:,1].plot(label='future', c='r')        
        plt.legend(loc='upper right')
        plt.xlabel('')
        plt.title('ver.'+ver+', '+future_model_name+ ', ' +df.columns[target_col] +'. history and future (out of the sample)'+ ', total time: '+model_training_time + ' sec')
        plt.savefig('Plot3a_history and future_'+future_model_name)
        plt.close()
    except Exception as e:
        print(e)
        
end_all=timeit.default_timer()
total_time=round(end_all-start_all,2)
print('-----------------------------')
print('total time:',total_time, 'sec')
#input('press any key')