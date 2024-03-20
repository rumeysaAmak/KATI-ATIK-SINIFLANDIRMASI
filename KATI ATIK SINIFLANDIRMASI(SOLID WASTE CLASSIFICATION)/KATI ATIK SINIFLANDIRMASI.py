# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:35:24 2024

@author: Lenovo
"""

import cv2
import urllib #URL'leri işlemek ve internet üzerindeki verilere erişmek için kullanılan bir Python modülüdür. İnternet üzerindeki verilere erişmek, dosyaları indirmek ve URL'lerle etkileşim kurmak için kullanılır.
import itertools #Python'da tekrarlayan veri yapıları üzerinde döngüler oluşturmak, kombinasyonlar oluşturmak, permütasyonlar yapmak ve veri yapılarını birleştirmek gibi işlemleri yapmak için kullanılan bir kütüphanedir.
import numpy as np #Python dilinde bilimsel hesaplamalar ve çok boyutlu diziler üzerinde işlemler yapmak için kullanılan temel bir kütüphanedir. Özellikle matris işlemleri, lineer cebir işlemleri ve rastgele sayı üretimi gibi işlemlerde yaygın olarak kullanılır
import pandas as pd #veri analizi ve veri manipülasyonu için kullanılan bir kütüphanedir. Veri okuma, filtreleme, gruplama, birleştirme ve dönüştürme gibi işlemleri kolaylaştırır. Veri çerçeveleri (DataFrame) üzerinde etkili bir şekilde çalışmayı sağlar.
import seaborn as sns #Python dilinde istatistiksel veri görselleştirmesi için kullanılan bir kütüphanedir. Matplotlib'e dayanır ve daha kolay kullanımı ve daha estetik görünümü ile bilinir. Veri görselleştirmesi yaparken sıklıkla kullanılır.
import random, os, glob #"random", rastgele sayılar üretmek ve rastgele öğeler seçmek için kullanılan bir kütüphanedir. Özellikle rastgele örnekleme ve rastgele sayı üretimi gibi işlemlerde kullanılır.
                        #"os", işletim sistemi işlemleri için kullanılan bir kütüphanedir. Dosya ve dizin işlemleri yapmak, sistemdeki değişkenleri kontrol etmek ve çalışma dizini gibi işlemleri gerçekleştirmek için kullanılır.
                        #"glob", dosya adı eşleştirme (dosya yollarını bir kalıpla eşleştirme) işlemleri için kullanılan bir modüldür. Belirli bir kalıba uyan dosya yollarını bulmak için kullanılır.
from imutils import paths #"imutils", görüntü işleme için yardımcı işlevler içeren bir kütüphanedir. Görüntü yeniden boyutlandırma, döndürme, kesme gibi işlemleri kolaylaştırır.
import matplotlib.pyplot as plt #"matplotlib.pyplot", grafik çizmek ve görselleştirmek için kullanılan bir kütüphanedir. Matplotlib'in bir alt modülüdür ve görsel sunumlar oluşturmak için sıklıkla kullanılır.
from sklearn.utils import shuffle #"sklearn.utils", Scikit-learn kütüphanesinin yardımcı işlevlerini içeren bir modüldür. Veri setlerini karıştırmak, parçalamak ve dönüştürmek gibi işlemleri gerçekleştirmek için kullanılır.
from urllib.request import urlopen #urllib.request Python'un standart kütüphanelerinden biridir ve ağ üzerinden URL'leri açmak ve okumak için kullanılır. urlopen() fonksiyonu, belirtilen bir URL'yi açmak ve bu URL üzerindeki veriye erişmek için kullanılır. Bu veri genellikle HTML sayfaları, resimler, ses dosyaları veya herhangi bir çevrimiçi erişilebilir veri kaynağı olabilir.

#warningleri kapatmak için kullanılan kütüphane
import warnings
warnings.filterwarnings('ignore')

#model değerlendirmek için kullanılan kütüphaneler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#model için kullanılan kütüphaneler
import tensorflow as tf #TensorFlow, makine öğrenimi ve derin öğrenme için açık kaynaklı bir platformdur. Grafiksel hesaplama için kullanılır ve özellikle yapay sinir ağları oluşturmak için yaygın olarak kullanılır.
from tensorflow.keras.models import Sequential #Keras, TensorFlow'da bulunan yüksek seviyeli bir derin öğrenme kütüphanesidir. Sequential modeli, sıralı bir sinir ağı oluşturmayı sağlar, katmanları ardışık olarak sıralar.
from tensorflow.keras.preprocessing import image #Görüntü verilerini ön işlemek için kullanılan bir alt modüldür. Görüntü verilerini yüklemek, dönüştürmek ve işlemek için işlevler içerir.
from tensorflow.keras.utils import to_categorical #Kategorik değişkenler için one-hot encoding işlemi yapar. Bu, sınıf etiketlerini kategorik veriye dönüştürmek için kullanılır.
from tensorflow.keras.callbacks import ModelCheckpoint #Model eğitimi sırasında belirli aralıklarla model ağırlıklarını kaydetmeyi sağlayan bir geri çağırma işlevi.
from tensorflow.keras.callbacks import EarlyStopping #Modelin aşırı uyumunu önlemek için eğitimi otomatik olarak durduran bir geri çağırma işlevi. Belirli bir ölçüm (genellikle doğruluk) belirli bir süre boyunca iyileşmezse eğitimi durdurur.
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D #Evrişimli sinir ağı (CNN) katmanlarını oluşturur. Görüntü işleme ve desen tanıma görevlerinde yaygın olarak kullanılır.
                                                                                                    #Evrişimli katmanlardan sonra düzleştirme işlemi yapar, yani 2D tensörleri düz bir vektöre dönüştürür.
                                                                                                    #Maksimum havuzlama işlemi uygular. Bu, evrişimli katmanlardan geçen özellik haritalarını özetlemek için kullanılır ve hesaplama maliyetini azaltırken özelliklerin öğrenilmesini korur.
                                                                                                    #Tam bağlantılı (fully connected) katmanlar oluşturur. Yapay sinir ağının nihai çıktılarını üretmek için kullanılır.
                                                                                                    #Ağın aşırı uyumunu önlemek için ağın rastgele bir bölümünü kapatır. Bu, genelleme yeteneğini artırır.
                                                                                                    #Giriş katmanındaki giriş özellik haritalarını korurken, giriş özelliklerinin rastgele bir bölümünü bırakır. Bu, evrişimli sinir ağlarında yaygın olarak kullanılır.
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img #Görüntü verilerini artırma ve ön işleme için kullanılan bir veri artırma sınıfı. Veri artırma, eğitim veri setinin çeşitliliğini artırarak modelin genelleme yeteneğini artırır.


#Bu kod, atık sınıflandırma modelinin eğitiminde kullanılacak olan veri kümesinin dizinlerini ve etiketlerini belirtir. Bu dizinlerdeki görüntüler, modelin eğitimi sırasında kullanılacak ve belirtilen etiketlerle birlikte eğitim verisi olarak kullanılacaktır.
dir_path = r'C:\Users\Lenovo\.spyder-py3\Garbage classification\Garbage classification' # Atık görüntülerinin bulunduğu dizini belirtir. Yani, bu dizinde her bir atık tipinin klasörü bulunmalıdır ve her bir klasörde ilgili atık tipine ait görüntüler bulunmalıdır.
target_size = (224, 224) #Görüntülerin yeniden boyutlandırılacağı hedef boyutu belirtir. Burada, (224, 224) piksel boyutlarına yeniden boyutlandırılacak.
waste_labels = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5} #Atık sınıflarını ve bunlara karşılık gelen etiketlerini içeren bir sözlüktür. Her atık türü bir anahtar olarak belirtilmiştir ve bunlara karşılık gelen değerler etiketlerdir. Örneğin, 'cardboard' için etiket 0, 'glass' için etiket 1, ve böyle devam eder.

def load_datasets(path): #Bu satır, load_datasets adında bir fonksiyon tanımlar. Bu fonksiyon, veri setini yüklemek için kullanılır ve bir yol (path) parametresi alır.
    #x = [] ve labels = []: x ve labels isimli boş liste oluşturulur. x, görüntülerin piksel verilerini tutmak için kullanılacak, labels ise her görüntünün etiketini saklayacak.
    x = []
    labels = []
    for label in os.listdir(path): #os.listdir(path) fonksiyonu, belirtilen dizin içindeki dosya ve klasörleri listeler. Bu döngü, verilen yolun içindeki her bir klasörü (label) dolaşır.
        label_path = os.path.join(path, label) #os.path.join() fonksiyonu, dizinler arasında doğru bir şekilde birleştirme yapar. Bu satır, her bir sınıf klasörünün tam yolunu oluşturur.
        for img_file in os.listdir(label_path): # İç içe geçmiş ikinci bir döngü, her bir sınıf klasörünün içindeki dosyaları (img_file) dolaşır.
            img_path = os.path.join(label_path, img_file) #os.path.join(label_path, img_file): Bu satır, her bir görüntünün tam yolunu oluşturur.
            img = cv2.imread(img_path) #cv2.imread() fonksiyonu, belirtilen dosyadan bir görüntüyü okur. Bu satır, görüntüyü okur ve img değişkenine atar.
            img = cv2.resize(img, target_size) #cv2.resize() fonksiyonu, bir görüntüyü belirtilen boyuta yeniden boyutlandırır. Bu satır, okunan görüntüyü target_size boyutuna yeniden boyutlandırır.
            
            #x.append(img) ve labels.append(waste_labels[label]): Yeniden boyutlandırılmış görüntü x listesine eklenir ve bu görüntünün etiketi labels listesine eklenir. Etiket, waste_labels sözlüğünden alınır.
            x.append(img) 
            labels.append(waste_labels[label])

    x, labels = shuffle(x, labels, random_state=42) #shuffle() fonksiyonu, x ve labels listelerini karıştırır. Bu, veri setinin rastgele sıralanmasını sağlar. random_state=42 parametresi, her çalıştırmada aynı rastgele sıralamanın elde edilmesini sağlar.


    #print(f"X boyutu: {np.array(x).shape}") ve print(f"Label sınıf sayısı: {len(np.unique(labels))}, gözlem sayısı: {len(labels)}"): Bu satırlar, yüklenen veri setinin boyutunu ve etiket sınıf sayısını ekrana yazdırır.
    print(f"X boyutu: {np.array(x).shape}")
    print(f"Label sınıf sayısı: {len(np.unique(labels))}, gözlem sayısı: {len(labels)}")

    return np.array(x), np.array(labels) # Fonksiyon, x ve labels listelerini Numpy dizilerine dönüştürerek bu dizileri döndürür.

x, labels = load_datasets(dir_path) #eri seti yüklenir ve x değişkenine görüntüler, labels değişkenine etiketler atanır.
input_shape = x[0].shape #Modelin giriş boyutunu belirler. x[0] ifadesi, veri setindeki birinci görüntüyü temsil eder ve shape özelliği, bu görüntünün boyutunu verir. Bu boyut, modelin giriş şekli olarak kullanılır.
print(input_shape)



# veri setinden örnekler gösterilmesi
def visualize_img(image_batch, label_batch): #Bu satır, visualize_img adında bir fonksiyon tanımlar. Bu fonksiyon, verilen görüntü ve etiket gruplarını görselleştirmek için kullanılır. İki parametre alır: image_batch (görüntü kümesi) ve label_batch (etiket kümesi).
    plt.figure(figsize=(10, 10)) #plt.figure() fonksiyonu, yeni bir figür oluşturur. figsize parametresi, figürün genişliğini ve yüksekliğini belirler. Burada, 10x10 bir inçlik bir figür oluşturulur.
    for i in range(25): #25 adet alt grafik oluşturmak için bir döngü başlatılır. Bu döngü, 5 satır ve 5 sütun şeklinde 25 adet alt grafik oluşturmak için kullanılır.
        plt.subplot(5, 5, i + 1) #lt.subplot() fonksiyonu, bir alt grafik oluşturur. İlk iki parametre, alt grafiklerin satır ve sütun sayısını belirler. Üçüncü parametre, oluşturulan alt grafiklerin indeksini belirler. İndeks 1'den başlar, bu yüzden i + 1 kullanılır.
        plt.imshow(image_batch[i]) #plt.imshow() fonksiyonu, bir görüntüyü görselleştirmek için kullanılır. image_batch[i], image_batch içindeki i indeksli görüntüyü temsil eder.
        plt.title(list(waste_labels.keys())[label_batch[i]]) #plt.title() fonksiyonu, grafik üzerine başlık ekler. Burada, başlık olarak kullanılacak etiketi belirlemek için waste_labels sözlüğündeki anahtarları (sınıf adlarını) kullanır. label_batch[i], i indeksli görüntünün etiketini belirtir.
        plt.axis("off")

visualize_img(x, labels) #visualize_img fonksiyonu, x ve labels değişkenlerini kullanarak görüntüleri ve etiketleri görselleştirir. Bu, veri setindeki görüntülerin bir alt kümesini görsel olarak incelememizi sağlar.
  


"""veriyi hazırlamak"""
#train veri seti için bir generator tanımlıyoruz
# Bu satır, ImageDataGenerator sınıfından bir nesne oluşturur. ImageDataGenerator, görüntü verilerini artırmak ve ön işlemek için kullanılan bir sınıftır.
train= ImageDataGenerator(horizontal_flip=True, #Görüntülerin yatay olarak ters çevrilmesini sağlar. Bu, veri artırma tekniklerinden biridir ve modelin genelleme yeteneğini artırabilir.
                          vertical_flip=True, #Görüntülerin dikey olarak ters çevrilmesini sağlar. Bu da bir veri artırma tekniğidir ve genellikle kullanılan yaygın bir dönüştürmedir.
                          validation_split=0.1, #Veri setinin bir bölümünü (yüzde olarak belirtilen kısmı) doğrulama için ayırır. Bu parametre, modelin eğitim sırasında ayrılan doğrulama veri setinin oranını belirler. Burada, verinin yüzde 10'u doğrulama için ayrılmıştır.
                          rescale=1./255, #Görüntü piksellerinin ölçeklendirilmesini sağlar. Bu, piksel değerlerini 0 ile 1 arasına ölçekler. Normalizasyon işlemidir ve modelin daha hızlı ve stabil öğrenmesine yardımcı olabilir.
                          shear_range=0.1, #Kesme (shear) etkisinin uygulanma aralığını belirler. Kesme, bir görüntünün yatay eksen boyunca kaydırılmasıdır. Bu, veri artırma yöntemlerinden biridir.
                          zoom_range=0.1, #Yakınlaştırma aralığını belirler. Bu, bir görüntünün yakınlaştırılmasını sağlar. Veri setinin çeşitliliğini artırmak için kullanılır.
                          width_shift_range=0.1, #Genişlik kaydırma aralığını belirler. Bu, bir görüntünün yatay olarak kaydırılmasını sağlar. Yine, veri artırma tekniği olarak kullanılır.
                          height_shift_range=0.1) #Yükseklik kaydırma aralığını belirler. Bu, bir görüntünün dikey olarak kaydırılmasını sağlar. Veri setinin çeşitliliğini artırmak için kullanılır.

"""test veri seti için bir generator tanımlıyoruz"""

test= ImageDataGenerator(rescale=1/255,
                         validation_split=0.1) #Bu kod, test veri kümesi için bir ImageDataGenerator nesnesi oluşturur. rescale=1/255, piksel değerlerini 0 ile 1 arasına ölçekler. Bu, görüntü piksel değerlerini 0 ile 255 arasında normalleştirir. validation_split=0.1, veri setini %90 eğitim ve %10 doğrulama kümeleri olarak bölerek doğrulama kümesinin ne kadarını oluşturacağını belirtir.

train_generator = train.flow_from_directory(dir_path,
                                            target_size=target_size,
                                            class_mode='categorical',
                                            subset='training') #Bu kod, eğitim veri seti için bir veri akışı üreticisi (ImageDataGenerator.flow_from_directory) oluşturur. dir_path dizinindeki görüntüleri yükler. target_size=target_size, yüklenen görüntülerin hedef boyutunu belirtir. class_mode='categorical', modelin çoklu sınıflandırma yapacağını belirtir. subset='training', veri akışının eğitim alt kümesini almasını sağlar.

test_generator = test.flow_from_directory(directory=dir_path,
                                          target_size=target_size,
                                          class_mode='categorical',
                                          subset='validation') #Bu kod, test veri seti için bir veri akışı üreticisi oluşturur. directory=dir_path, görüntüleri yükleyeceği dizini belirtir. target_size=target_size, yüklenen görüntülerin hedef boyutunu belirtir. class_mode='categorical', modelin çoklu sınıflandırma yapacağını belirtir. subset='validation', veri akışının doğrulama alt kümesini almasını sağlar.


"""modelleme"""
#sıfırdan CNN modeli kurma
#sequential
#convolution layer, conv2D
#havuzlama katmanı(pooling layer)
#activation layer
#flattening katmanı
#dense katmanı
#dropout katmanı

model=Sequential() #Bu satır, bir Keras modeli oluşturur. Sequential() fonksiyonu, ardışık (sıralı) bir sinir ağı modeli oluşturmayı sağlar.

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(input_shape),activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=(2,2))) #Bu satır, bir maksimum havuzlama katmanı (MaxPooling2D) ekler. pool_size=2, havuzlama penceresinin boyutunu belirtir. strides=(2,2), havuzlama penceresinin kaydırma adımlarını belirtir.

#Bu işlemi iki kez daha tekrarlayarak modeldeki evrişimli katman ve maksimum havuzlama katmanlarının sayısını arttırır.

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')) #Bu satır, bir evrişimli katman (Conv2D) ekler. filters=32, katmanda kullanılacak filtre sayısını belirtir. kernel_size=(3,3), filtrelerin boyutunu belirtir. padding='same', giriş ve çıkış boyutlarını aynı yapmak için kenar dolgusu ekler. input_shape=(input_shape), modelin giriş şeklini belirtir. activation='relu', ReLU aktivasyon fonksiyonunu kullanır.
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Flatten()) #Bu satır, düzleştirme katmanı ekler. Düzleştirme katmanı, evrişimli ve havuzlama katmanlarından gelen çıktıları düz bir vektöre dönüştürür. Bu, tam bağlantılı (fully connected) katmanlara giriş olarak kullanılabilir.

model.add(Dense(units=64,activation='relu')) # Bu satır, tam bağlantılı bir gizli katman (Dense) ekler. units=64, gizli katmandaki nöron sayısını belirtir. activation='relu', ReLU aktivasyon fonksiyonunu kullanır.
model.add(Dropout(rate=0.2)) #Bu satır, ağın aşırı uyumu önlemek için dropout (bırakma) katmanı ekler. rate=0.2, her bir ağırlık katmanındaki bağlantıların rastgele olarak %20'sini kapatır.
#tekrar ederek bir gizli katman ve dropout katmanı daha ekler
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=6,activation='softmax')) #Bu satır, çıkış katmanını ekler. units=6, çıkış katmanındaki nöron sayısını belirtir. Veri setinde altı sınıf olduğu için bu değer altıdır. activation='softmax', softmax aktivasyon fonksiyonunu kullanarak çıktıları olasılık dağılımına dönüştürür. 

#Bu, çok sınıflı sınıflandırma problemleri için yaygın bir uygulamadır.


"""MODEL ÖZETİ"""
model.summary() # Bu satır, modelin özetini yazdırır. Modelin katmanlarını, her katmandaki parametre sayısını, çıktı şekillerini ve toplam parametre sayısını gösterir. Bu özet, modelin mimarisini ve katmanların bağlantılarını görselleştirmeye yardımcı olur.

"""Optimizasyon ve Değerlendirme Metriklerinin Ayarlanmas"""
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),"acc"]) #Bu satır, modelin derlenmesini sağlar. Derleme işlemi, modelin eğitimi sırasında kullanılacak kayıp fonksiyonunu, optimize ediciyi ve değerlendirme metriklerini belirler.
                                                                                      #loss='categorical_crossentropy': Çok sınıflı sınıflandırma problemleri için yaygın bir kayıp fonksiyonu olan çapraz entropiyi belirtir.
                                                                                      #optimizer='adam': Adam optimize ediciyi kullanır. Adam, adaptif moment tahmini algoritmasıdır ve gradient tabanlı optimize ediciler arasında yaygın olarak kullanılır.
                                                                                      #metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"]: Modelin performansını değerlendirmek için kullanılacak metrikleri belirtir. Burada, hassasiyet (Precision), duyarlılık (Recall) ve doğruluk (Accuracy) metrikleri kullanılmıştır.

callbacks=[EarlyStopping(monitor='val_loss',patience=50, verbose=1, mode="min"),
           
           ModelCheckpoint(filepath='mymodel.h5',monitor='val_loss',mode="min",save_best_only=True,save_weight_only=False)] #callbacks: Bu kısım, model eğitim sırasında kullanılacak geri çağırma işlevlerini belirtir. Geri çağırma işlevleri, eğitim sırasında belirli olaylar gerçekleştiğinde (örneğin, belirli bir metriğin iyileşmediği durumda eğitimi durdurma gibi) belirli eylemleri gerçekleştirmek için kullanılır.
                                                                                                                            #EarlyStopping: Eğitimi, belirli bir metriğin (burada 'val_loss') belirli bir süre boyunca iyileşmediği durumda durdurur. patience parametresi, eğitimi durdurmadan önce kaç epoch boyunca iyileşme olmadığını belirtir.
                                                                                                                            #ModelCheckpoint: Eğitim sırasında model ağırlıklarını ve model mimarisini kaydeder. 
                                                                                                                            #filepath parametresi, kaydedilecek model dosyasının adını belirtir. 
                                                                                                                            #save_best_only=True, sadece doğrulama kaybının en düşük olduğu epoch'larda modelin kaydedileceğini belirtir. Bu, modelin doğrulama performansının en iyisi olduğu noktada modelin kaydedilmesini sağlar.

"""MODELİN EĞİTİLMESİ"""
#Eğitim işlemi başladığında, her bir epoch için eğitim ve doğrulama işlemi gerçekleştirilir. Her epoch sonunda, belirtilen metrikler kaydedilir ve eğitim sürecinin geçmişi (history nesnesi) oluşturulur. Bu geçmiş nesnesi, modelin eğitim sürecinin istatistiklerini (örneğin, kayıp ve doğruluk) içerir ve daha sonra eğitim sonuçlarını analiz etmek için kullanılabilir.

history= model.fit_generator(generator=train_generator,
                             epochs=100,
                             validation_data=test_generator,
                             workers=4,
                             steps_per_epoch=2276//32,
                             validation_steps=251//32)
#generator=train_generator: Eğitim veri seti için bir veri akışı üreticisi belirtilir. 
#train_generator, önceden belirlenmiş parametrelerle oluşturulmuş bir veri akışıdır ve eğitim için kullanılacak veri kümesini temsil eder.
#epochs=100: Eğitim için kaç epoch (iterasyon) yapılacağını belirtir. Bir epoch, modelin tamamıyla eğitim veri setini geçmesi anlamına gelir.
#validation_data=test_generator: Doğrulama için kullanılacak veri akışı belirtilir. test_generator, önceden belirlenmiş parametrelerle oluşturulmuş bir veri akışıdır ve modelin eğitimi sırasında doğrulama için kullanılacak veri kümesini temsil eder.
#workers=4: Eğitim sırasında kullanılacak işçi sayısını belirtir. İşçiler, veri akışından verileri alarak eğitim işlemini hızlandırmak için kullanılır.
#steps_per_epoch=2276//32: Her bir epoch için kaç adımın alınacağını belirtir. // operatörü, tam bölme işlemini gerçekleştirir. Bu durumda, eğitim veri setinde toplam 2276 görüntü bulunmaktadır ve bir adımda 32 görüntü işlenecektir. Bu, bir epoch'taki adım sayısını belirler.
#validation_steps=251//32: Doğrulama sırasında kaç adımın alınacağını belirtir. // operatörü, tam bölme işlemini gerçekleştirir. Bu durumda, doğrulama veri setinde toplam 251 görüntü bulunmaktadır ve bir adımda 32 görüntü işlenecektir. Bu, doğrulama sırasındaki adım sayısını belirler.

"""ACCURACY VE LOSS GRAFİKLERİ"""

#Accuracy grafiği

plt.figure(figsize=(20,5)) #Bu satır, yeni bir figür oluşturur ve figürün boyutunu (genişlik ve yükseklik) belirler. figsize=(20, 5) parametresi, genişliği 20 inç, yüksekliği 5 inç olan bir figür oluşturur.
plt.subplot(1,2,1) #Bu satır, 1 satır ve 2 sütunlu bir alt grafik düzeni içindeki birinci alt grafik bölgesini seçer.
plt.plot(history.history['acc'],label='Training Accuracy') #Bu satır, eğitim süreci boyunca elde edilen eğitim doğruluğunu (acc) çizgi grafiği olarak çizer. label='Training Accuracy' parametresi, grafikteki çizginin etiketini belirtir.
plt.plot(history.history['val_acc'],label='Validation Accuracy') #u satır, eğitim süreci boyunca elde edilen doğrulama doğruluğunu (val_acc) çizgi grafiği olarak çizer. label='Validation Accuracy' parametresi, grafikteki çizginin etiketini belirtir.
plt.legend(loc='lower right') #Bu satır, grafiğe bir lejant (açıklama) ekler. Lejant, grafikteki çizgilerin neyi temsil ettiğini gösterir. loc='lower right' parametresi, lejantın sağ alt köşede görüntüleneceğini belirtir.
plt.xlabel('Epoch', fontsize=16) #Bu satır, x ekseni etiketini belirler. fontsize=16 parametresi, etiketin font boyutunu ayarlar.
plt.ylabel('Accuracy', fontsize=16) #Bu satır, y ekseni etiketini belirler. fontsize=16 parametresi, etiketin font boyutunu ayarlar.
plt.ylim([min(plt.ylim()),1]) #Bu satır, y ekseni aralığını belirler. Y ekseni aralığı, [0, 1] aralığında olacak şekilde ayarlanır.
plt.title('Training and Validation Accuracy', fontsize=16) #Bu satır, grafiğe başlık ekler. fontsize=16 parametresi, başlığın font boyutunu ayarlar.


#loss grafiği

plt.subplot(1,2,2) #Bu satır, 1 satır ve 2 sütunlu bir alt grafik düzeni içindeki ikinci alt grafik bölgesini seçer.
plt.plot(history.history['loss'],label='Training Loss') #Bu satır, eğitim süreci boyunca elde edilen eğitim kaybını (loss) çizgi grafiği olarak çizer. label='Training Loss' parametresi, grafikteki çizginin etiketini belirtir.
plt.plot(history.history['val_loss'],label='Validation Loss') #Bu satır, eğitim süreci boyunca elde edilen doğrulama kaybını (val_loss) çizgi grafiği olarak çizer. label='Validation Loss' parametresi, grafikteki çizginin etiketini belirtir.
plt.legend(loc='upper right') #Bu satır, grafiğe bir lejant (açıklama) ekler. Lejant, grafikteki çizgilerin neyi temsil ettiğini gösterir. loc='upper right' parametresi, lejantın sağ üst köşede görüntüleneceğini belirtir.
plt.xlabel('Epoch', fontsize=16) #Bu satır, x ekseni etiketini belirler. fontsize=16 parametresi, etiketin font boyutunu ayarlar.
plt.ylabel('Loss', fontsize=16) #Bu satır, y ekseni etiketini belirler. fontsize=16 parametresi, etiketin font boyutunu ayarlar.
plt.ylim([0,max(plt.ylim())]) #Bu satır, y ekseni aralığını belirler. Y ekseni aralığı, 0'dan başlayarak en yüksek kayıp değerine kadar olan aralığa ayarlanır.
plt.title('Training and Validation Loss', fontsize=16) #Bu satır, grafiğe başlık ekler. fontsize=16 parametresi, başlığın font boyutunu ayarlar.
plt.show() # Bu satır, oluşturulan grafikleri görüntüler.


"""MODEL BAŞARI DEĞERLENDİRMESİ(EVALUATION)"""

loss,precision, recall, acc= model.evaluate(test_generator, batch_size=32)

print("\nTest accuracy: %.lf%%"%(100.0*acc))
print("\nTest loss: %.lf%%"%(100.0*loss))
print("\nTest precision: %.lf%%"%(100.0*precision))
print("\nTest recall: %.lf%%"%(100.0*recall))


"""CLASSIFICATION REPORT"""

x_test, y_test=test_generator.next() #Bu satır, test veri kümesinden bir sonraki veri yığınını (x_test) ve karşılık gelen etiketleri (y_test) elde etmek için test_generator.next() işlevini kullanır. Test veri kümesi, test_generator veri akışı nesnesi üzerinden üretilir.

y_pred=model.predict(x_test) #Bu satır, modeli kullanarak test veri kümesindeki görüntülerin tahminlerini yapar. model.predict() yöntemi, görüntü yığınını (x_test) alır ve her bir görüntü için model tarafından yapılan tahminleri içeren bir dizi üretir (y_pred).

y_pred=np.argmax(y_pred,axis=1) # Bu satır, tahmin edilen sınıfları one-hot kodlamadan (y_pred) gerçek sınıf etiketlerine dönüştürür. np.argmax() işlevi, her bir tahmin vektöründe en yüksek olasılığa sahip sınıfın indeksini döndürür. axis=1 parametresi, her bir tahmin vektörü için en büyük değerin satır bazında indeksini döndürür.

y_test=np.argmax(y_test,axis=1) #Bu satır, gerçek sınıf etiketlerini one-hot kodlamadan (y_test) gerçek sınıf etiketlerine dönüştürür. np.argmax() işlevi, gerçek sınıf etiketlerindeki en yüksek olasılığa sahip sınıfın indeksini döndürür. axis=1 parametresi, her bir etiket vektörü için en büyük değerin satır bazında indeksini döndürür.


target_names=list(waste_labels.keys()) #Bu satır, sınıf etiketlerini (waste_labels sözlüğünün anahtarları) bir liste (target_names) olarak alır. Bu, sınıf isimlerini daha sonra sınıflandırma raporunda kullanmak için yapılır.

print(classification_report(y_test,y_pred,target_names=target_names)) #Bu satır, scikit-learn kütüphanesinin classification_report() fonksiyonunu kullanarak sınıflandırma raporunu yazdırır. Bu rapor, modelin doğruluk, hassasiyet, duyarlılık ve F1-score gibi performans ölçütlerini her bir sınıf için ayrı ayrı gösterir. target_names parametresi, sınıf isimlerini (etiketlerini) belirtir ve raporun daha anlaşılır olmasını sağlar.


"""confusion matrix"""

cm=confusion_matrix(y_test, y_pred) #Bu satır, gerçek etiketler (y_test) ve tahmin edilen etiketler (y_pred) kullanılarak bir karışıklık matrisi oluşturur. Karışıklık matrisi, modelin her bir sınıf için ne kadar doğru veya yanlış tahmin yaptığını gösteren bir tablodur.

#Bu satır, bir plot_confusion_matrix adında bir işlev tanımlar. Bu işlev, bir karışıklık matrisini görselleştirmek için kullanılır ve çeşitli parametreler alır:
#cm: Görselleştirilecek karışıklık matrisi.
#classes: Sınıf etiketlerinin bir listesi.
#normalize: Matrisin normalize edilip edilmeyeceğini belirten bir boolean değer. Varsayılan olarak False olarak ayarlanmıştır.
#title: Görselin başlığı.
#cmap: Renk haritası. Varsayılan olarak mavi tonlarında bir renk haritası (plt.cm.Blues) kullanılır.
def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm= cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis] # Eğer normalize parametresi True ise, karışıklık matrisi normalize edilir. Bu, her sınıfın doğru ve yanlış sınıflandırma oranlarını görmemizi sağlar.
        
    plt.figure(figsize=(8,6)) #Yeni bir figür oluşturur ve figürün boyutunu (genişlik ve yükseklik) belirler.
    plt.imshow(cm,interpolation='nearest',cmap=cmap) #Karışıklık matrisini görselleştirir. interpolation='nearest' parametresi, piksel değerlerini görüntülerken yakınsama algoritması kullanır. cmap=cmap parametresi, belirtilen renk haritasını kullanır.
    plt.title(title) #Görselin başlığını belirler.
    plt.colorbar() #Renk skalasını (color bar) görselin yanına ekler. Renk skalanın altındaki sayılar, karışıklık matrisindeki piksel değerlerini gösterir.
    tick_marks=np.arange(len(classes)) #Sınıf etiketleri için işaretler oluşturur.
    plt.xticks(tick_marks,classes, rotation=45) #x ekseni üzerindeki etiketleri belirler. rotation=45 parametresi, etiketlerin 45 derece döndürülmesini sağlar.
    plt.yticks(tick_marks,classes) #y ekseni üzerindeki etiketleri belirler.
    fmt= '.2f' if normalize else 'd' #Format belirleyici oluşturur. Eğer matris normalize edilmişse, ondalık format kullanılır.
    thresh= cm.max()/2.  #Bir eşik değeri belirler. Bu eşik değeri, metnin beyaz veya siyah renkte olacağı bir sınırlayıcı belirler.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])): #İki for döngüsü, matrisin her bir elemanını dolaşır.
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black") #Her bir matris hücresine, hücredeki değeri yazan metin ekler. Bu metin, hücrenin ortasına hizalanır. Eğer hücrenin değeri eşik değerinden büyükse, metin beyaz renkte olur; aksi takdirde siyah renkte olur.
        plt.tight_layout() #Alt grafiklerin sıkı bir şekilde düzenlenmesini sağlar.
        plt.ylabel('True label', fontweight="bold") #y ekseni etiketini belirler.
        plt.xlabel('Predicted label', fontweight="bold") # x ekseni etiketini belirler.

plot_confusion_matrix(cm, waste_labels.keys(),
                      title='confusion matrix',
                      cmap=plt.cm.OrRd) #Karışıklık matrisini görselleştirmek için plot_confusion_matrix işlevini çağırır. waste_labels.keys() parametresi, sınıf etiketlerini belirtir ve matrisin sınıf etiketlerini doğru şekilde etiketlemesini sağlar. title='confusion matrix' parametresi, görselin başlığını belirtir. cmap=plt.cm.OrRd parametresi, renk haritasını belirler.


"""MODELİN KULLANILMASI/TEST EDİLMESİ"""

waste_labels={0:'cardboard', 1:'glass',2:'metal',3:'paper',4:'plastic',5:'trash'} #Sınıf etiketlerini ve bunlara karşılık gelen indeksleri içeren bir sözlük oluşturulur. Bu sözlük, modelin tahmin ettiği sınıf indekslerini gerçek sınıf etiketlerine dönüştürmek için kullanılır.

def model_testing(path): #model_testing adında bir işlev tanımlanır. Bu işlev, bir görüntünün sınıfını tahmin etmek için kullanılır.path parametresi: Görüntünün dosya yolunu alır.
    img= image.load_img(path,target_size=(target_size)) #Verilen dosya yolundan bir görüntü yükler ve hedef boyuta yeniden boyutlandırır. target_size parametresi, görüntünün yeniden boyutlandırılacağı hedef boyutu belirtir.
    img= image.img_to_array(img,dtype=np.uint8) #Yüklü görüntüyü bir NumPy dizisine dönüştürür. Dönüştürülen dizi, dtype=np.uint8 parametresiyle belirtilen veri türünde olacaktır.
    img=np.array(img)/255.0 #Görüntüyü 0 ile 1 arasında değerlere ölçekler. Bu, görüntüyü normalize etmek için yapılır.
    p=model.predict(img.reshape(1,224,224,3)) #Modeli kullanarak görüntünün sınıfını tahmin eder. Görüntü, modelin beklediği giriş biçimine ((1, 224, 224, 3)) yeniden şekillendirilir.
    predicted_class= np.argmax(p[0]) #Tahmin edilen sınıfın indeksini belirler. np.argmax() işlevi, en yüksek olasılığa sahip sınıfın indeksini döndürür.
    
    return img, p, predicted_class #Fonksiyon, görüntüyü, tahmin olasılıklarını ve tahmin edilen sınıf indeksini döndürür.


#İlk üç satır, üç farklı görüntü dosyasının yolunu alır ve model_testing fonksiyonunu çağırır. Bu fonksiyon, görüntüyü alır, model üzerinde tahminleme yapar ve tahmin sonuçlarını döndürür. Bu üç satır, üç farklı görüntü için bu işlemi gerçekleştirir.
img1,p1,predicted_class1=model_testing(r'C:\Users\Lenovo\.spyder-py3\metal.jpg')
img2,p2,predicted_class2=model_testing(r'C:\Users\Lenovo\.spyder-py3\cam.jpg')
img3,p3,predicted_class3=model_testing(r'C:\Users\Lenovo\.spyder-py3\atik-kagit.jpg')

plt.figure(figsize=(20,60)) #Bir figür oluşturur ve bu figürün boyutunu belirler. figsize=(20,60) parametresiyle, genişlik 20 birim, yükseklik 60 birim olarak ayarlanır.


#plt.subplot(141), plt.subplot(142), plt.subplot(143): Birinci satırın ilk, ikinci ve üçüncü sütunlarını oluşturur. Yani bu kodlar, tek bir satırda üç farklı görüntüyü gösterecek olan üç farklı subplot oluşturur. Her subplot, farklı bir görüntünün gösterileceği yere karşılık gelir.
#plt.imshow(img1.squeeze()), plt.imshow(img2.squeeze()), plt.imshow(img3.squeeze()): Her subplot içinde, imshow fonksiyonu kullanılarak görüntüler gösterilir. squeeze() işlevi, bir görüntü dizisini boyutlarından sıkıştırır ve eksenlerin sayısını azaltır, böylece imshow fonksiyonu doğru şekilde görüntüleyebilir.
#plt.title(): Her subplot için bir başlık belirler. Başlık, görüntünün maksimum olasılığı ve tahmin edilen sınıfı içerir. str(np.max(p1[0],axis=-1)) ifadesi, maksimum olasılığı hesaplar. waste_labels[predicted_class1] ifadesi, tahmin edilen sınıfın etiketini belirler.
plt.subplot(141)
plt.axis('off')
plt.imshow(img1.squeeze())
plt.title("maximum probablity: "+str(np.max(p1[0],axis=-1))+"\n"+"predicted class: "+ str(waste_labels[predicted_class1]))
plt.imshow(img1);

plt.subplot(142)
plt.axis("off")
plt.imshow(img2.squeeze())
plt.title("maximum probablity: "+str(np.max(p2[0],axis=-1))+"\n"+"predicted class: "+ str(waste_labels[predicted_class2]))
plt.imshow(img2);

plt.subplot(143)
plt.axis("off")
plt.imshow(img3.squeeze())
plt.title("maximum probablity: "+str(np.max(p3[0],axis=-1))+"\n"+"predicted class: "+ str(waste_labels[predicted_class3]))
plt.imshow(img3);

#Bu kodlar, üç farklı görüntünün model tarafından tahmin edilmiş sınıfını ve tahmin edilen sınıfın maksimum olasılığını görsel olarak gösterir.








                          
                          
                          
                          
                          
                          
                          
