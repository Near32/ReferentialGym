# Obverter Approach vs Straight-Through Gumbel-Softmax Approach

This example investigate the behavioural differences between the Obverter approach and the Straight-Through Gumbel-Softmax approach in the context of a descriptive discriminative referential game with 3 distractors.

## How to launch?

### Obverter:
```
./train_obverter_discriminative.sh <SEED> <DESCRIPTIE_RATIO> <SENTENCE_LENGTH> <VOCAB_SIZE> <NBR_DISTRACTORS> 
```

e.g.:

```
./train_obverter_discriminative.sh 10 0.8 20 5 3 
```

### ST-GS:
```
./train_stgs_discriminative.sh <SEED> <DESCRIPTIE_RATIO> <SENTENCE_LENGTH> <VOCAB_SIZE> <NBR_DISTRACTORS> 
```

