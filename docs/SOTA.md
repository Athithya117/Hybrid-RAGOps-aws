



elastic/multilingual-e5-small-optimized



1. shavarani/SpEL(base only) for SOTA english 

2. **mReFinED** for fastest , supports 9 languages: Arabic, English, Spanish, German, Farsi, Japanese, Tamil, Turkish, and possibly Lithuanian.

3. fallback to wannaphong/BELA  from meta that supports
```sh
af  Afrikaans                 am  Amharic                   ar  Arabic
arg Aragonese                ast Asturian                 az  Azerbaijani
bar Bavarian                 be  Belarusian               bg  Bulgarian
bn  Bengali                  br  Breton                   bs  Bosnian
ca  Catalan                  ckb Central Kurdish         cs  Czech
cy  Welsh                    da  Danish                   de  German
el  Greek                    en  English                  eo  Esperanto
es  Spanish                  et  Estonian                 eu  Basque
fa  Persian/Farsi            fi  Finnish                  fr  French
fy  Western Frisian          ga  Irish                    gan Gan Chinese
gl  Galician                 gu  Gujarati                 he  Hebrew
hi  Hindi                    hr  Croatian                 hu  Hungarian
hy  Armenian                 ia  Interlingua              id  Indonesian
is  Icelandic                it  Italian                  ja  Japanese
jv  Javanese                 ka  Georgian                 kk  Kazakh
kn  Kannada                  ko  Korean                   ku  Kurdish
la  Latin                    lb  Luxembourgish            lt  Lithuanian
lv  Latvian                  mk  Macedonian               ml  Malayalam
mn  Mongolian                mr  Marathi                  ms  Malay
my  Burmese                  nds Low German              ne  Nepali
nl  Dutch                    nn  Norwegian (Nynorsk)      no  Norwegian
oc  Occitan                  pl  Polish                   pt  Portuguese
ro  Romanian                 ru  Russian                  scn Sicilian
sco Scots                   sh  Serbo-Croatian           si  Sinhala
sk  Slovak                   sl  Slovenian                sq  Albanian
sr  Serbian                  sv  Swedish                  sw  Swahili
ta  Tamil                    te  Telugu                   th  Thai
tl  Tagalog                  tr  Turkish                  tt  Tatar
uk  Ukrainian                ur  Urdu                     uz  Uzbek
vi  Vietnamese               war Waray                    wuu Wu Chinese
yi  Yiddish                  zh  Modern Standard Chinese  zh_classical Classical Chinese
zh_yue Cantonese
```


For ultra-high throughput, consider batching requests, using ONNX optimized pipelines, or running quantized inference — these can further boost speed while retaining accuracy


Run this model using GGUF quantization via optimum or ctransformers to get sub-10 ms latency per query even on budget CPUs



These two together form a compact but accurate pipeline suitable for **graph+vector-based RAG** at production scale.






### **Qdrant Payload (Final JSONL Format)**

This is the full, enriched payload per vector chunk to be indexed in Qdrant:

```sh




```
