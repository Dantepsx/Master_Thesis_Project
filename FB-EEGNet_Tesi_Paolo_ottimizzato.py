# IMPORT DELLE LIBRERIE

import numpy as np
from scipy.io import loadmat
import os # PERMETTE DI INTERAGIRE CON FILE E CARTELLE
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "6"
# In questo modo permetto a tensorflow di parallelizzare le operazioni
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time
import seaborn as sns # SERVE PER LA LEGGIBILITà DELLA MATRICE DI CONFUSIONE
import json # permette di salvare dizionari Python in un file testuale leggibile
import random
from scipy.signal import butter, filtfilt
import tempfile
import shutil

# CARICAMENTO DEL DATASET

def load_dataset(path):

    subjects = []
    labels = []
    eeg_data = []

    # Voglio estrarre dalle cartelle dei soggetti i segnali EEG, e per farlo devo scorrere prima sui soggetti e poi sui vari
    # cicli, che ho chiamato trials

    for subject in os.listdir(path):
        subject_path = os.path.join(path, subject)   # Questo comando costruisce una stringa che rappresenta il percorso
        if not os.path.isdir(subject_path): # Questo invece verifica che il l'elemento sia una cartella, altrimenti lo salta
            continue

        for trial in os.listdir(subject_path):
            trial_path = os.path.join(subject_path, trial)
            if not os.path.isdir(trial_path):
                continue

            for file in os.listdir(trial_path):
                if file.startswith("EEG"):                      # Se il file inizia con EEG lo prendo in considerazione
                    file_path = os.path.join(trial_path, file)

                    f = loadmat(file_path)  # loadmat restituisce un dizionario, non il segnale eeg vero e proprio, quindi 

                    # bisogna svolgere altre operazioni

                    var_keys = [key for key in f.keys() if not key.startswith('__')]

                # Il dizionario è formato da varie chiavi, come timestamp ecc, che iniziano tutte con __; quindi, se prendo le 
                # chiavi di quelli che non iniziano con __, ottengo l'array di segnale EEG

                    if len(var_keys) == 1:  # Voglio che ci sia un solo elemento nelle chiavi trovate, quello relativo all'EEG
                        key = var_keys[0]
                        eeg = f[key]  # Prendo il vettore eeg dal dizionario f

                        eeg_data.append(eeg)

                        # Ora devo estrarre la frequenza associata ad ogni file EEG e salvarla come label

                        freq = int(file.split('_')[1].split('.')[0])
                        labels.append(freq)

                        # Il file EEG si chiama EEG_freq.mat, allora spezzo questa stringa in un vettore [EEG, freq.mat], e
                        # prendo il secondo elemento, freq.mat, e lo spezzo in [freq, mat], e poi mi prendo freq

                        subjects.append(subject)

    return eeg_data, np.array(labels), np.array(subjects)

    # Perché non abbiamo usato una struttura più nidificata (soggetto -> ciclo -> segnale EEG) invece di una struttura piatta
    # (soggetti, label, segnali sono vettori indipendenti)? Perché in un approccio di deep learning è più facile 
    # computazionalmente ricevere un vettore di campioni piuttosto che ogni volta dover accedere alla struttura nidificata

# Funzione per ottenere le 3 bande per FB-EEGNet

def bandpass_filter(lowcut, highcut, fs, order = 4):
    
    nyquist = 0.5*fs # Ho bisogno di definirla perché i filtri digitali lavorano con frequenze normalizzate
    low = lowcut/nyquist
    high = highcut/nyquist
    
    return butter(order, [low,high], btype = 'band') # Costruisco il filtro con i suoi coefficienti

# Funzione per filtrare i campioni per ogni soggetto e per ogni frequenza

def filter_subjects(X,Y, subjects,target_subjects):
    idx = [i for i, n in enumerate(subjects) if n in target_subjects]

    # subjects = [s1,s1,s1...s30,s30,s30]; enumerate mi restituisce un vettore di tuple, [(0,s1),(1,s1)...], ogni soggetto si 
    # ripete 5x8 = 40 volte, e voglio gli indici solo relativi ai target, in modo da prendere gli indici relativi solo
    # a certi segnali EEG

    return X[idx], Y[idx], subjects[idx]

# Funzione per ottenere i dati dei 3 rami e prepararli per Keras con normalizzazione Z-score

def prepare_fb(EEG_fb, sub_train, sub_val, sub_test, labels, subjects, epsilon = 1e-8):
    
    # Devo applicare filter_subjects ad ogni banda
    # E normalizzare su ognuna di esse
    # epsilon è un valore piccolo che evita la divisione per 0
    
    result_train, result_val, result_test = [], [], []
    y_train = y_val = y_test = None
    sub_train_out = sub_val_out = sub_test_out = None
    
    for i, band in enumerate(EEG_fb):
        
        X_tr , y_tr ,sub_tra_samples = filter_subjects(band, labels, subjects, sub_train)
        X_v  , y_vl ,sub_val_samples = filter_subjects(band, labels, subjects, sub_val)
        X_te , y_te ,sub_tes_samples = filter_subjects(band, labels, subjects, sub_test)
        
        if i == 0: # y e samples sono identici per tutte le bande, basta salvarli una volta
            y_train, y_val, y_test = y_tr, y_vl, y_te
            sub_train_out, sub_val_out, sub_test_out = sub_tra_samples, sub_val_samples, sub_tes_samples
            
        mean = X_tr.mean(axis = 0, keepdims = True)
        std = X_tr.std(axis = 0, keepdims = True)
        
        result_train.append(np.expand_dims(((X_tr - mean)/(std+epsilon)).astype(np.float32), axis=-1))
        result_val.append(  np.expand_dims(((X_v  - mean)/(std+epsilon)).astype(np.float32), axis=-1))
        result_test.append( np.expand_dims(((X_te - mean)/(std+epsilon)).astype(np.float32), axis=-1))
    return result_train, result_val, result_test, y_train, y_val, y_test, sub_train_out, sub_val_out, sub_test_out

    # con np.expand e astype preparo i dati per Keras,
    # aggiungendo una dimensione relativa al canale
    # axis = -1 serve proprio ad aggiungere la dimensione in più che vuole keras
    
# Definizione FB-EEGNet
    
def build_FBEEGNet(channels, samples, num_classes, hyperparameters, num_bands):
    
    inputs = []
    branch_outputs = []
    
    for _ in range(num_bands): # Un ramo per ogni banda
        inp = keras.Input(shape=(channels, samples, 1))
        inputs.append(inp)
        
        # A differenza di EEGNet non posso creare model.Sequential, dato che
        # Ho bisogno di 3 rami separati per le 3 bande. Allora dico a keras crea una sotto rete 
        # con queste dimensioni, la appendo alla lista input altrimenti si perde. 
        # Dopodichè, costruisco mano mano il singolo ramo, come se fosse una lista di c++ e scorressi la lista tramite nodi. 
        # Alla fine appendo il ramo alla lista di output altrimenti viene perso. 
        # Dopodiche, fondo i 3 rami ed aggiungo il dense finale, come da struttura di FB-EEGNET
        
        x = layers.Conv2D(
            hyperparameters["F1"], (1, hyperparameters["kernel_length"]),
            padding = "same", use_bias = False,
            kernel_initializer=initializers.RandomNormal(0,0.01),
            kernel_regularizer=regularizers.l2(hyperparameters["l2"])
        )(inp)
        
        x = layers.BatchNormalization()(x)
        
        x = layers.DepthwiseConv2D(
        (channels, 1), depth_multiplier = hyperparameters["D"], padding='valid',
        depthwise_constraint = max_norm(1.), use_bias=False,
        depthwise_initializer = initializers.RandomNormal(0,0.01),
        depthwise_regularizer = regularizers.l2(hyperparameters["l2"])
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AvgPool2D((1, 4))(x)
        x = layers.Dropout(hyperparameters["dropout_rate"])(x)
        
        x = layers.SeparableConv2D(hyperparameters["F2"], (1,16), padding='same', use_bias=False, 
            pointwise_initializer = initializers.RandomNormal(0, 0.01), depthwise_initializer = initializers.RandomNormal(0, 0.01),
            depthwise_regularizer=regularizers.l2(hyperparameters["l2"]),
            pointwise_regularizer=regularizers.l2(hyperparameters["l2"]))(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AvgPool2D((1, 8))(x)
        x = layers.Dropout(hyperparameters["dropout_rate"])(x)
        
        x = layers.Flatten()(x)
        branch_outputs.append(x)
        
        # Fondo poi i 3 rami
    
    merged = layers.Concatenate()(branch_outputs)
    
    out = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=initializers.RandomNormal(0, 0.01),
        kernel_regularizer=regularizers.l2(hyperparameters["l2"])
    )(merged)
    
    model = keras.Model(inputs = inputs, outputs = out)
    model.compile(
        optimizer = Adam(hyperparameters["learning_rate"], clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
        run_eagerly=False# La loss è sparse, quindi anche l'accuracy deve esserlo, altrimenti Keras si aspetta dei vettori one hot
        # con la sparse_categorical_crossentropy ho label intere, non one hot encoding
        # Recall = Tp/(Tp + Fn) ; Precision = Tp/(Tp + Fp)
    )

    return model

# DEFINISCO LA FUNZIONE PER IL CALCOLO DELLE METRICHE E DELL'U_INTRA_SOGGETTO AD OGNI FOLD

# Per ogni soggetto ho 40 EEG, divisi in cicli da 5, quindi 8 per volta. su 8 di questi calcolo la metrica, la calcolo 5 volte e poi faccio la media
# X_test ha dimensioni 40x8x1136 40 segnali, 8 righe e 1136 colonne, ogni segnale ha una frequenza, i primi 8 segnali sono il primo ciclo con le
# 8 frequenze, ci sono 8 righe perché 8 sono gli elettrodi utilizzati

def mean_metric(model,X_test_fb,y_test):
    
    acc_list = []
    rec_list = []
    prec_list = []
    num_cycles = 5
      
   
    for i in range(num_cycles):
       
        start = 8*i
        end = 8*(i+1)
       
        # Devo indicizzare ogni banda separatamente, e poi costruire la lista
        X_cycle = [band[start:end] for band in X_test_fb] 
        
        # Qui sto inserendo, per il primo ciclo ad esempio, i campioni da
        # 0 a 8 della banda 1, della banda 2 e della 3. Il modello riceve
        # i 3 array simultaneamente perché devono scorrere in parallelo
        
        y_cycle = y_test[start:end]
       
        y_pred_prob = model.predict(X_cycle, verbose = 0) # Calcola la probabilità che il segnale EEG appartenga ad una delle 8 classi
    
        # Ha dimensioni 40 x 8 (nel caso di questa funzione 5x8): ho 40 matrici (nel caso della funzione 5) del soggetto di test in ingresso, 
        # ognuna che rappresenta un segnale EEG ad una data frequenza;
        # Per ognuna di queste matrici devo dire se appartiene ad una delle 8 classi, quindi vi associo una riga che contiene 8 probabilità
        # di appartenenza ad una determinata classe
    
        y_pred = np.argmax(y_pred_prob, axis = 1) # Prende il valore massimo, quello che molto probabilmente è la classe corretta
    
        # axis = 1 indica che sto cercando il massimo muovendomi per le colonne, quindi sto controllando riga per riga, ottengo così un vettore 40 x 1
        # (5x1) Uso argmax e non max poiché non mi interessa il valore massimo ma la classe a cui appartiene quel valore massimo
       
        test_acc = accuracy_score(y_cycle, y_pred)
        test_recall = recall_score(y_cycle, y_pred, average = 'macro', zero_division = 0)  # zero_division serve per settare 0 se il denominatore è 0
        test_precision = precision_score(y_cycle, y_pred, average = 'macro', zero_division = 0)
        
        # macro lo devo mettere perché se non lo metto recall_score mette di default binary, e quindi pensa che sia un problema di classificazione
        # binario, quando in realtà non lo è e quindi va in crisi. macro di base serve per rendere le classi bilanciate e quindi rendere il calcolo
        # di queste metriche più corretto nel caso in cui le classi siano sbilanciate; non è questo il caso ma senza macro il programma non funziona.
       
        acc_list.append(test_acc)
        rec_list.append(test_recall)
        prec_list.append(test_precision)
    
    accuracy = np.mean(acc_list)
    recall = np.mean(rec_list)
    precision = np.mean(prec_list)
    
    u_intra = np.std(acc_list)/np.sqrt(num_cycles)

    return accuracy, recall, precision, u_intra

def make_dataset(X_fb, y, batch_size, shuffle=False): # Questa funzione mi permette di convertire i dati da array numpy in 
    # una pipeline ottimizzata per tensorflow, che gestisce i dati in modo più efficiente
    ds = tf.data.Dataset.from_tensor_slices((tuple(X_fb), y))
    
    # Fa si che ogni elemento del vettore X_fb sia una tupla, (insieme ad un elemento di y)
    # Questo rende più facile a tensorflow lavorare con i dati
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y))
        
    # Sul training lo shuffle è attivo: i campioni vengono mescolati ad ogni epoca per evitare che il modello impari l'ordine con cui arrivano i dati 
    # e quindi generalizzare meglio
    #Su validation e test lo shuffle è disattivato, non mi serve più
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Con ds.batch raggruppo i campioni in batch, batch che ottengo dall'iperparametro batch size
# Ogni volta che alleno una configurazione

#  mentre la CPU sta elaborando il batch N, il prefetch sta già preparando il batch N+1 in background
# AUTOTUNE fa sì che Tenso flow calibri autonomamente i batch

path = r"C:\Users\Utente\Desktop\Tesi Casaretta\DATASET_SSVEP_AR"
EEG_raw, labels_raw, subjects = load_dataset(path)

base_path = os.path.dirname(path) # prende il percorso della cartella superiore rispetto al path
results_root = os.path.join(base_path, "results") # costruisco il percorso result
os.makedirs(results_root, exist_ok=True) # serve a creare la cartella; se esiste già, non fa niente (exist_ok = True)

existing_runs = [
    d for d in os.listdir(results_root)
    if d.startswith("run_")
] # vede tutti i file "run" creati dentro results, serve per capire quanti ce ne sono

run_id = len(existing_runs) + 1
run_name = f"run_{run_id:03d}" # aggiungo il numero davanti al nome del file run
run_path = os.path.join(results_root, run_name) # costruisce il percorso per la nuova cartella run
os.makedirs(run_path) # crea la nuova cartella run

print(f"Salvataggio risultati in: {run_path}")
print(f'numero di segnali EEG: {len(EEG_raw)}')
print(f'Labels: ' , np.unique(labels_raw))

# EEG_raw contiene tutte le registrazioni EEG, ognuna delle quali ha dimensione 9 x N_samples, questo perché sono stati usati
# 8 elettrodi (9 righe, ma la prima sono tutti 0). ho usato la funzione np.unique per far vedere singolarmente
# le frequenze i soggetti visto che nei vettori labels_raw e subjects le frequenze ed i soggetti si ripetono.

 # PRE-PROCESSING

 # Devo prima rendere i segnali EEG della stessa lunghezza:

fs = 250
t_window = 1.25

l_min = int(fs*t_window)

# cancello la riga superiore e accorcio gli EEG alla lunghezza minima

eeg_8 = [eeg[1:,:] for eeg in EEG_raw]
EEG = np.array([eeg[:,:l_min] for eeg in eeg_8])

bands = [(6,60), (16,60), (24,60)]
num_bands = len(bands)
filter_coeffs = [bandpass_filter(low,high,fs) for (low,high) in bands] # Ottengo i coefficienti dei filtri per ogni banda

EEG_fb = []

for filter_coeff in filter_coeffs:
    
    b, a = filter_coeff
    EEG_band = filtfilt(b, a, EEG, axis = -1) # Applico il filtro sui dati; axis = -1 perchè l'ultima dimensione è il tempo e devo filtrare su quella
    EEG_fb.append(EEG_band)
    
    # Ottengo un vettore in cui ho i vari segnali EEG, tutte le registrazioni, separate in banda 1, banda 2, banda 3
    
EEG_fb = np.array(EEG_fb)

# Ora normalizzo le frequenze, in modo che vadano da 0 a 7

labels = np.array(labels_raw - 8)
subjects = np.array(subjects)

unique_subjects = list(np.unique(subjects))
print(unique_subjects)
num_folds = len(unique_subjects)
loso_results = [] # Definisco un vettore dove mettere i risultati: per ogni fold avrò un dizionario contenente il soggetto di test e le metriche

# DEFINIZIONE delle configurazioni di iperparametri

num_config = 80

num_classes=len(np.unique(labels))  # Come parametri iniziali scelgo quelli del paper di FB-EEGNet
channels = eeg_8[0].shape[0]
samples = EEG.shape[2] # EEG è un vettore di matrici, col primo indice accedo alla singola matrice, poi al numero di righe e poi al numero di colonne
dropoutRate = [0.4, 0.5, 0.6]
kernLength = int(fs/2)
F1 = [16, 32, 64] # 96 CORRISPONDE AL NUMERO DI BANDE PASSATE, 96 SONO TROPPE BANDE, OTTENGO INFORMAZIONI TROPPO SPECIFICHE
D = [1, 2]
learning_rate = [1e-3, 5*1e-4]
batch_size = [32, 64, 96]
num_Epochs = 200
l2 = [1e-3, 1e-4]

hyperparameters_all_config = [] # VETTORE CHE CONTIENE TUTTE LE CONFIGURAZIONI DI IPERPARAMETRI
config_sets = set()
hyper_root = os.path.join(run_path, "hyperparameters all configurations") # costruisco il percorso hyper nella cartella run
os.makedirs(hyper_root, exist_ok=True) # serve a creare la cartella per gli iperparametri; se esiste già, non fa niente (exist_ok = True)

while len(hyperparameters_all_config) < num_config:
    
    F1_choice = random.choice(F1)
    D_choice = random.choice(D)
    
    config_set = (
        random.choice(learning_rate),
        random.choice(dropoutRate),
        random.choice(batch_size),
        num_Epochs,
        F1_choice,
        D_choice,
        F1_choice*D_choice,
        kernLength,
        random.choice(l2),
        num_classes,
        channels,
        samples
    )
    
    if config_set in config_sets: # Questo serve a vedere se una configurazione randomica è già stata considerata; se si, salto i passaggi successivi ed il ciclo
        continue # riparte dall' inizio
    
    config_sets.add(config_set)
    
    hyperparameters = {
        "learning_rate": config_set[0],
        "dropout_rate": config_set[1],
        "batch_size": config_set[2],
        "epochs": config_set[3],
        "F1": config_set[4],
        "D": config_set[5],
        "F2" : config_set[6],
        "kernel_length": config_set[7],
        "l2" : config_set[8],
        "num_classes": num_classes,
        "channels": channels,
        "samples": samples
    }
    
    hyperparameters_all_config.append(hyperparameters)

    with open(os.path.join(hyper_root, f"hyperparameters{len(hyperparameters_all_config)}.json"), "w") as f: # costruisce il percorso del file che voglio 
        json.dump(hyperparameters, f, indent=4) # creare dentro la cartella run corrente; w = modalità scrittura, with serve per far chiudere il file dopo la scrittura
             
        # Scrivo le varie configurazioni in una cartella
print(num_classes)
print(channels)
print(samples)
    
tempo_tot_add = 0

u_intra_all = [] # Vettore che contiene tutte le accuracy intra-soggetto, mi serve dopo per fare la media

for fold, test_subject in enumerate(unique_subjects):
    
    sub_train_val = [s for s in unique_subjects if s != test_subject]
    
# Ora devo dividere i soggetti in training, validation e test

    sub_train , sub_val = train_test_split(
        sub_train_val,
        test_size = 0.2, random_state = 42
   )
    
    # Random state = 42 serve ad avere sempre gli stessi soggetti nei vettori di training e validation ad ogni ciclo for, serve per far si che non cambino

    X_train_fb, X_val_fb, X_test_fb, y_train, y_val, y_test, sub_train_samples, sub_val_samples, sub_test_samples = prepare_fb(
    EEG_fb, sub_train, sub_val, test_subject, labels, subjects
    )
    #  INIZIO PARTE DI CODICE DIVERSA DA PRIMA
    
    batch_sizes = list({h["batch_size"] for h in hyperparameters_all_config})

    train_ds_map = {
        bs: make_dataset(X_train_fb, y_train, bs, shuffle=True)
        for bs in batch_sizes
    }
    val_ds_map = {
        bs: make_dataset(X_val_fb, y_val, bs, shuffle=False)
        for bs in batch_sizes
    }
    
    # Qui trasformo i miei array in tuple per tensorflow: lo faccio fuori da ciclo degli iperparametri
    # dato che ogni configurazione avrà il suo batch size, è inutile farlo 80 volte, quindi lo faccio fuori
    # creo quindi un dizionario che ha come chiave il batch size, e quindi quando richiamerò i dati di training
    # lo farò solo una volta in funzione del batch size

    temp_dir = tempfile.mkdtemp()  # cartella temporanea gestita dal sistema
    
    # Questa cartella temporanea mi serve perché alleno 80 modelli in fase 1
    # 20 in fase 2, 5 in fase 3, ma ogni volta che alleno una configurazione
    # cancello il modello per avere più spazio, ma devo salvare i pesi da qualche parte
    # e li salvo nella cartella temporanea

    best_val_loss = np.inf
    best_config   = None
    best_ckpt_path = None  # percorso del checkpoint del vincitore

    start = time.time()

    # FASE 1 — 80 configurazioni, 20 epoche, patience 5
    
    print("Fase 1: 80 configurazioni, 20 epoche")
    fase1_results = []

    for config_id, hyperparameters in enumerate(hyperparameters_all_config):
        ckpt_path = os.path.join(temp_dir, f"cfg_{config_id}.weights.h5")
        
        # Ogni configurazione ha il suo file di checkpoint dove salvo i pesi,
        # Tutti all'interno della cartella temporanea

        model = build_FBEEGNet(channels, samples, num_classes, hyperparameters, num_bands)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5,
                        restore_best_weights=True, min_delta=1e-4),
            ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                            save_best_only=True, save_weights_only=True, verbose=0)
        ]
        
        # Model checkpoint mi serve per vedere se la loss è migliorata rispetto
        # all'iterazione precedente; se si, sovrascrive i pesi nel file della configurazione
        # nella cartella temporanea, (save_best_only), e salvo solo i pesi
        # (save_weihts_only) e non tutto il modello, che tanto posso ricostruire con 
        # build_FBEEGNet

        train_ds = train_ds_map[hyperparameters["batch_size"]]
        val_ds   = val_ds_map[hyperparameters["batch_size"]]

        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=20, callbacks=callbacks, verbose=0)

        min_val_loss = min(history.history["val_loss"])
        fase1_results.append({
            "config_id":      config_id,
            "hyperparameters": hyperparameters,
            "val_loss":       min_val_loss,
            "ckpt_path":      ckpt_path      # salvo il percorso del checkpoint
        })

        del model
        tf.keras.backend.clear_session()
        print(f"  Config {config_id+1}/80 - val_loss: {min_val_loss:.4f}")

    fase1_results.sort(key=lambda x: x["val_loss"])
    
    # Ordino le loss in una classifica
    # In pratica sto dicendo di confrontare le configurazioni in funzione
    # della loss
    
    top20 = fase1_results[:20]
    
    # Prendo solo i primi 20 modelli, cioè quelli migliori
    
    print(f"Fase 1 ok. Migliore val_loss: {fase1_results[0]['val_loss']:.4f}")

    # FASE 2, top 20, riparte dai pesi di fase 1, fino a 100 epoche, patience 10
    
    print("Fase 2: 20 configurazioni, ripresa da pesi fase 1")
    fase2_results = []

    for entry in top20:
        hyperparameters = entry["hyperparameters"]
        ckpt_path       = entry["ckpt_path"]        # stesso file, verrà sovrascritto se migliora
        config_id       = entry["config_id"]

        model = build_FBEEGNet(channels, samples, num_classes, hyperparameters, num_bands)
        model.load_weights(ckpt_path)               # la configurazione riparte con i pesi
        
        # Con cui aveva finito in fase 1

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                        restore_best_weights=True, min_delta=1e-4),
            ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                            save_best_only=True, save_weights_only=True, verbose=0)
        ]

        train_ds = train_ds_map[hyperparameters["batch_size"]]
        val_ds   = val_ds_map[hyperparameters["batch_size"]]

        history = model.fit(train_ds, validation_data=val_ds,
                            initial_epoch = 20,
                            epochs=120, callbacks=callbacks, verbose=0)
        # Keras continua dal punto in cui era, ma epochs=100 significa
        # 100 epoche aggiuntive rispetto a quelle già fatte in fase 1

        min_val_loss = min(history.history["val_loss"])
        fase2_results.append({
            "config_id":       config_id,
            "hyperparameters": hyperparameters,
            "val_loss":        min_val_loss,
            "ckpt_path":       ckpt_path
        })

        del model
        tf.keras.backend.clear_session()

    fase2_results.sort(key=lambda x: x["val_loss"])
    top5 = fase2_results[:5]
    print(f"Fase 2 ok. Migliore val_loss: {fase2_results[0]['val_loss']:.4f}")

    # FASE 3, top 5, riparte dai pesi di fase 2, fino a 200 epoche, patience 20
    
    print("Fase 3: 5 configurazioni, ripresa da pesi fase 2")
    fase3_results = []

    for entry in top5:
        hyperparameters = entry["hyperparameters"]
        ckpt_path       = entry["ckpt_path"]
        config_id       = entry["config_id"]

        model = build_FBEEGNet(channels, samples, num_classes, hyperparameters, num_bands)
        model.load_weights(ckpt_path)               # ← riparte da dove aveva finito la fase 2

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20,
                        restore_best_weights=True, min_delta=1e-4),
            ModelCheckpoint(filepath=ckpt_path, monitor='val_loss',
                            save_best_only=True, save_weights_only=True, verbose=0)
        ]

        train_ds = train_ds_map[hyperparameters["batch_size"]]
        val_ds   = val_ds_map[hyperparameters["batch_size"]]

        history = model.fit(train_ds, validation_data=val_ds,
                            initial_epoch = 120,
                            epochs=200, callbacks=callbacks, verbose=0)

        min_val_loss = min(history.history["val_loss"])
        fase3_results.append({
            "config_id":       config_id,
            "hyperparameters": hyperparameters,
            "val_loss":        min_val_loss,
            "ckpt_path":       ckpt_path,
            "history":         history.history      # salvo history solo qui
        })

        del model
        tf.keras.backend.clear_session()

    # FASE finale, in cui prendo la configurazione migliore e la do al soggetto di test
    fase3_results.sort(key=lambda x: x["val_loss"])
    best        = fase3_results[0]
    best_ckpt_path = best["ckpt_path"]

    best_model = build_FBEEGNet(channels, samples, num_classes,
                                best["hyperparameters"], num_bands)
    best_model.load_weights(best_ckpt_path)         # carica i pesi migliori del vincitore
    shutil.rmtree(temp_dir)
    
    # Cancello la cartella temporanea, non mi serve più

    best_config = {
    "hyperparameters": best["hyperparameters"],
    "loss":       [float(x) for x in best["history"]["loss"]],
    "val_loss":   [float(x) for x in best["history"]["val_loss"]],
    "accuracy":   [float(x) for x in best["history"]["sparse_categorical_accuracy"]],
    "val_accuracy": [float(x) for x in best["history"]["val_sparse_categorical_accuracy"]]
    }
    
    # Qui scrivo float(x) perché quando accedo alla history della best_config
    # i valori sono liste di oggetti numpy.float32, e json non sa come lavorarci
    # Quindi otterrei un errore; di conseguenza, devo convertire tutto in float
    
    # FINE Parte di codice nuova

    tempo_add = time.time() - start
    tempo_tot_add += tempo_add 
    test_acc, test_recall, test_precision, u_intra = mean_metric(best_model, X_test_fb, y_test)
    u_intra_all.append(u_intra)
    
    loso_results.append({
        'SOGGETTO': test_subject,
        'accuracy': test_acc,
        'recall': test_recall,
        'precision': test_precision
    })
    
    # u_intra_all.append(u_intra)
    y_pred_prob = best_model.predict(X_test_fb, verbose = 0) # Calcola la probabilità che il segnale EEG appartenga ad una delle 8 classi
    
    # Ha dimensioni 40 x 8: ho 40 matrici del soggetto di test in ingresso, ognuna che rappresenta un segnale EEG ad una data frequenza;
    # Per ognuna di queste matrici devo dire se appartiene ad una delle 8 classi, quindi vi associo una riga che contiene 8 probabilità
    # di appartenenza ad una determinata classe
    
    y_pred = np.argmax(y_pred_prob, axis = 1) # Prende il valore massimo, quello che molto probabilmente è la classe corretta
    
    # axis = 1 indica che sto cercando il massimo muovendomi per le colonne, quindi sto controllando riga per riga, ottengo così un vettore 40 x 1
    # Uso argmax e non max poiché non mi interessa il valore massimo ma la classe a cui appartiene quel valore massimo
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True, # Scrive i numeri dentro le celle
        fmt='d', # serve per avere numeri interi
        cmap='Blues', # definisce il tipo di feature map
        xticklabels=np.unique(labels),
        yticklabels=np.unique(labels)
    )
    plt.title(f'Confusion Matrix – {test_subject}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    subject_dir = os.path.join(run_path, test_subject) # Costruisce il percorso del soggetto dentro la cartella run
    os.makedirs(subject_dir, exist_ok=True) # crea la cartella

    # Metriche soggetto
    subject_metrics = {
        "accuracy": float(test_acc),
        "recall": float(test_recall),
        "precision": float(test_precision),
        "u_intra": float(u_intra)
    }

    with open(os.path.join(subject_dir, "metrics.json"), "w") as f: # costruisce il percorso del file metriche e le scrive nella cartella
        json.dump(subject_metrics, f, indent=4)

    # Confusion matrix
    plt.savefig(os.path.join(subject_dir, f'cm_{test_subject}.png'), dpi=300, bbox_inches='tight')
    # salva la matrice di confusione nella cartella del soggetto
    plt.close()
    
    # Salva configurazione migliore
    with open(os.path.join(subject_dir, f"best_configuration_{test_subject}.json"), "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(test_subject)

all_accuracies = [diz['accuracy'] for diz in loso_results]
acc_med = np.mean(all_accuracies)
u_inter = np.std(all_accuracies)/np.sqrt(len(unique_subjects))
u_intra_all_sq = np.square(u_intra_all)
u_tot = np.sqrt(u_inter**2 + np.mean(u_intra_all_sq)) # Calcolo dell'incertezza totale
    
summary = {
    "acc_med": float(acc_med),
    "u_inter": float(u_inter),
    "u_tot": float(u_tot)
}

with open(os.path.join(run_path, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Tempo totale di addestramento
with open(os.path.join(run_path, "training_time.txt"), "w") as f:
    f.write(f"Tempo totale di addestramento (s): {tempo_tot_add:.2f}\n")

# Con questi 2 comandi creo il percorso file e inserisco anche u_inter, u_tot ed il tempo di addestramento