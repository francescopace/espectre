# ESPectre - Advanced Feature Extraction

## Overview

Sistema avanzato di feature extraction per rilevamento movimento basato su CSI, implementato con metodi puramente matematici (non ML). Il sistema estrae **17 feature** dai dati CSI e usa un **detection score pesato** per identificare movimenti.

## Feature Implementate (17 totali)

### 1. Time-Domain Features (6 feature)
- **Mean**: Media delle ampiezze CSI
- **Variance**: Varianza delle ampiezze ⭐ (ALTAMENTE DISCRIMINANTE: 94→398)
- **Skewness**: Asimmetria della distribuzione
- **Kurtosis**: Curtosi (eventi impulsivi vs continui)
- **Entropy**: Entropia di Shannon
- **IQR**: Interquartile range ⭐ (DISCRIMINANTE: 10→18)

### 2. Spatial Features (3 feature)
- **Spatial Variance**: Varianza spaziale tra sottoportanti
- **Spatial Correlation**: Correlazione di Pearson
- **Spatial Gradient**: Tasso di cambiamento spaziale ⭐ (ALTAMENTE DISCRIMINANTE: 8.3→19.8)

### 3. Temporal Features (3 feature)
- **Autocorrelation Lag-1**: Pattern periodici
- **Zero-Crossing Rate**: Frequenza attraversamento media
- **Peak Rate**: Tasso di picchi locali

### 4. Multi-Window Analysis (3 feature)
- **Variance Short** (~1s): Movimenti rapidi ⭐ (ECCELLENTE: 0.002→0.054, +2400%)
- **Variance Medium** (~5s): Pattern intermedi
- **Variance Long** (~10s): Trend generali

### 5. Derivative Features (2 feature)
- **First Derivative**: Tasso di cambiamento
- **Second Derivative**: Accelerazione

## Detection Multi-Criterio

### Pesi Ottimizzati (Configurabili Runtime)
Basati su analisi empirica delle feature più discriminanti:

- **Variance**: 35% (feature più importante)
- **Spatial Gradient**: 30% (molto discriminante)
- **Variance Short**: 25% (eccellente per movimenti rapidi)
- **IQR**: 10% (buono)

**Totale**: 100%

### Range di Normalizzazione
- Variance: 0-400 (ottimizzato per sensibilità)
- Spatial Gradient: 0-25
- Variance Short: 0-0.1 (massima sensibilità)
- IQR: 0-25

## Filtri Avanzati (Sprint 2 - COMPLETATO)

### Filtro di Hampel (Outlier Removal)
Rimuove outlier usando il metodo MAD (Median Absolute Deviation).

**Funzionamento**:
- Calcola mediana e MAD della finestra (5 campioni)
- Identifica outlier: `|valore - mediana| > threshold × MAD × 1.4826`
- Sostituisce outlier con la mediana

**Comandi CLI**:
```bash
# Abilita filtro Hampel
./espectre-cli.sh hampel_filter on

# Configura threshold (1.0-10.0, default: 3.0)
./espectre-cli.sh hampel_threshold 3.0

# Disabilita
./espectre-cli.sh hampel_filter off
```

**MQTT**:
```bash
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"hampel_filter","enabled":true}'
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"hampel_threshold","value":3.0}'
```

### Filtro Savitzky-Golay (Smoothing)
Smoothing polinomiale che preserva le caratteristiche del segnale.

**Parametri**:
- Window size: 5 campioni
- Ordine polinomiale: 2
- Coefficienti pre-calcolati: [-0.0857, 0.3429, 0.4857, 0.3429, -0.0857]

**Comandi CLI**:
```bash
# Abilita filtro Savitzky-Golay
./espectre-cli.sh savgol_filter on

# Disabilita
./espectre-cli.sh savgol_filter off
```

**MQTT**:
```bash
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"savgol_filter","enabled":true}'
```

### Normalizzazione Adattiva
Normalizzazione Z-score con statistiche running (sempre attiva).

**Funzionamento**:
- Calcola media e varianza running con algoritmo di Welford
- Adaptation rate: 0.01 (adattamento lento)
- Z-score: `(valore - media) / stddev`

**Statistiche**:
```bash
# Visualizza stato filtri e statistiche normalizer
./espectre-cli.sh filters
```

Output:
```json
{
  "hampel_enabled": false,
  "hampel_threshold": 3.0,
  "savgol_enabled": false,
  "savgol_window_size": 5,
  "normalizer_mean": 0.234,
  "normalizer_variance": 0.045,
  "normalizer_samples": 1523
}
```

### Pipeline di Filtraggio
I filtri sono applicati in sequenza ottimale:

```
CSI Raw Data
    ↓
normalize_variance()
    ↓
apply_filters():
  1. Hampel filter (se abilitato) → rimuove outlier
  2. Savitzky-Golay (se abilitato) → smoothing
  3. Adaptive normalizer → aggiorna statistiche
    ↓
History Buffer
    ↓
Feature Extraction (17 features)
    ↓
Detection Score (multi-criterio pesato)
```

### Quando Usare i Filtri

**Hampel Filter**:
- ✅ Ambiente con interferenze sporadiche
- ✅ Presenza di spike nei dati CSI
- ✅ Falsi positivi da outlier
- ❌ Movimenti molto rapidi (potrebbe rimuoverli)

**Savitzky-Golay**:
- ✅ Segnale rumoroso ma con pattern regolari
- ✅ Ridurre jitter nelle transizioni
- ✅ Migliorare stabilità detection
- ❌ Movimenti molto rapidi (introduce lag)

**Best Practice**:
1. Inizia con filtri disabilitati
2. Analizza i dati con `./espectre-cli.sh features`
3. Se vedi molti spike → abilita Hampel
4. Se vedi molto rumore → abilita Savitzky-Golay
5. Monitora l'impatto con `./espectre-cli.sh monitor`

## Comandi CLI Completi

### Visualizzare Feature Estratte
```bash
./espectre-cli.sh features
```

### Visualizzare e Modificare Pesi
```bash
# Mostra pesi correnti
./espectre-cli.sh weights

# Modifica pesi runtime (via MQTT)
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"weight_variance","value":0.4}'
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"weight_spatial_gradient","value":0.3}'
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"weight_variance_short","value":0.2}'
mosquitto_pub -h broker -t "topic/cmd" -m '{"cmd":"weight_iqr","value":0.1}'
```

### Stati Granulari
```bash
# Abilita 5 stati (IDLE, MICRO, DETECTED, INTENSE)
./espectre-cli.sh granular_states on

# Disabilita (torna a 2 stati: IDLE, DETECTED)
./espectre-cli.sh granular_states off
```

### Analisi e Calibrazione
```bash
# Analizza dati raccolti e suggerisci threshold
./espectre-cli.sh analyze

# Imposta threshold
./espectre-cli.sh threshold 0.38
```

## Stato Implementazione

### ✅ Sprint 1: Feature Extraction Avanzato (COMPLETATO)
- 17 feature matematiche implementate
- Integrazione CSI callback
- Comandi MQTT e CLI

### ✅ Sprint 2: Filtri Avanzati (COMPLETATO)
- Filtro Hampel (outlier removal)
- Filtro Savitzky-Golay (smoothing)
- Normalizzazione adattiva
- Comandi MQTT e CLI per controllo filtri

### ✅ Sprint 3: Detection Multi-Criterio (COMPLETATO)
- Sistema punteggio pesato
- Pesi configurabili runtime
- State machine granulare (5 stati)
- Ottimizzazione basata su analisi empirica

### 🔄 Prossimi Sviluppi

#### Sprint 4: Calibrazione e Ottimizzazione
- Calibrazione multi-fase
- Adattamento continuo baseline
- Parallelizzazione dual-core

#### Sprint 5: Testing e Tuning
- Test scenari reali
- Metriche performance
- Fine-tuning parametri

## Performance

### Overhead Stimato
- **CPU**: +15-20% con feature extraction + filtri
- **RAM**: +10KB SRAM + ~2KB stack temporaneo
- **Latency**: +8-12ms per ciclo di processing

### Ottimizzazioni Implementate
- Algoritmi O(n) o O(n log n)
- Uso principalmente stack (no heap persistente)
- Solo IQR usa heap temporaneo (~100 bytes)
- Gestione robusta errori e casi limite

## Note Tecniche

1. **Feature extraction sempre attiva**: Sistema usa detection multi-criterio
2. **Filtri disabilitati di default**: Nessun impatto su performance se non abilitati
3. **Configurazione runtime**: Tutti i parametri modificabili via MQTT
4. **Modularità**: Ogni componente indipendente
5. **Robustezza**: Gestione divisione per zero e casi limite

## Riferimenti

Implementazione basata su:
- RGANet paper (PubMed 2025)
- Analisi spettrale CSI
- Parametri fisici del canale
- Metodi di fingerprinting matematico
