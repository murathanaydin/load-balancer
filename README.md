# Softmax Action Selection Tabanlı Yük Dengeleyici (Load Balancer) Simülasyonu


## 🎯 Proje Amacı ve Kapsamı

Projenin amacı, performansları zamanla değişen (**non-stationary**) ve anlık dalgalanmalar gösteren (**noisy**) sunuculardan oluşan bir kümede (cluster), toplam bekleme süresini (latency) minimize etmektir.
Klasik statik algoritmalar (örneğin Round-Robin veya Random), sunucuların anlık performans durumlarını dikkate almaz. Bu projede, Pekiştirmeli Öğrenme (Reinforcement Learning) literatüründe sıkça kullanılan **Softmax Action Selection** algoritması uygulanarak, sunucuların performans geçmişine dayalı olasılıksal ve dinamik bir yönlendirme mekanizması tasarlanmıştır.

### Ortam (Environment) Özellikleri
- **Non-stationary (Durağan Olmayan):** Sunucuların ortalama yanıt süreleri (latency) sabit değildir, zamanla rastgele kaymalar (drift) yaşar.
- **Noisy (Gürültülü):** Gelen her yanıt, ortalama değerin etrafında Gaussian bir gürültüye (standart sapmaya) sahiptir.

## ⚙️ Teknik Altyapı ve Algoritma

### Softmax Algoritması
Sistem, Exploration (Keşif) ve Exploitation (Sömürü) dengesini kurmak için Softmax formülünü kullanır. Düşük gecikme süresi (latency) hedeflediğimiz için, Q-değerleri (tahmini gecikmeler) negatif ödül olarak sisteme verilir.


### 🛡️ Nümerik Stabilite (Numerical Stability) Çözümü
Softmax fonksiyonundaki üstel hesaplamalar, özellikle büyük değerlerde bilgisayar belleğinde **Overflow (Taşma)** hatasına yol açar. Bu projede problemi çözmek için **Shift-Invariance** tekniği kullanılmıştır:

Üstel fonksiyona girmeden önce dizideki en büyük değer, tüm elemanlardan çıkarılır. Bu matematiksel müdahale olasılık dağılımını kesinlikle değiştirmez, ancak değerleri 0 ve negatif aralığa çekerek e^0 = 1 stabiliteyi garanti altına alır ve programın çökmesini engeller.

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. Repoyu klonlayın:
   ```bash
   git clone <sizin-repo-linkiniz>
   cd <repo-klasoru>

2. Gerekli kütüphaneleri yükleyin:
Bash
pip install numpy matplotlib

3. Simülasyonu başlatın:
Bash
python main.py

📊 Çıktılar ve Analiz
Program çalıştığında terminal üzerinde her sunucu için öğrenilmiş son Q-Değerlerini (tahmini latency) gösterir ve ardından bir performans analizi grafiği (softmax_analysis.png) üretir. Grafik, algoritmanın zaman ilerledikçe sistemi öğrenerek ortalama gecikme süresini nasıl optimize ettiğini kümülatif olarak sergiler.
