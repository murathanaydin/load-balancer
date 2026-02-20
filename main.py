import numpy as np
import matplotlib.pyplot as plt


# --- 1. ORTAM (ENVIRONMENT) ---
class Server:
    """
    Non-stationary (zamanla değişen) ve Noisy (gürültülü) sunucu simülasyonu.
    """

    def __init__(self, server_id):
        self.id = server_id
        # Başlangıçta rastgele bir ortalama yanıt süresi (50ms - 150ms arası)
        self.mean_latency = np.random.randint(50, 150)
        print(f"Server {server_id} başlangıç ortalama latency: {self.mean_latency} ms")

    def respond(self):
        # 1. Non-stationary Özellik: Her adımda performans biraz kayar (Drift)
        drift = np.random.normal(0, 1.5)
        self.mean_latency += drift
        self.mean_latency = max(10, self.mean_latency)  # Minimum 10ms sınırı

        # 2. Noisy Özellik: Gürültülü yanıt (Standart sapma yüksek)
        actual_latency = np.random.normal(self.mean_latency, 10.0)
        return max(1, actual_latency)  # 1ms'den az olamaz


# --- 2. AGENT (SOFTMAX LOAD BALANCER) ---
class SoftmaxLoadBalancer:
    def __init__(self, n_servers, temperature=20.0):
        self.n_servers = n_servers
        self.temperature = temperature

        # Q-Values: Her sunucu için tahmin edilen ortalama latency
        # Başlangıçta 0 veya ortalama bir değer verebiliriz.
        self.q_values = np.zeros(n_servers)
        self.counts = np.zeros(n_servers)  # Hangi sunucuyu kaç kez seçtik

    def select_server(self):
        """
        Softmax Action Selection ile sunucu seçer.
        Nümerik Stabilite için Shift-Invariance tekniği uygulanır.
        """
        # Biz Latency'i minimize etmek istiyoruz.
        # Softmax genelde ödülü (reward) maximize eder.
        # Bu yüzden Latency'nin negatifini alarak "Reward"a çeviriyoruz.
        # Düşük latency = Yüksek Reward (negatif olarak 0'a daha yakın)
        rewards = -1 * self.q_values

        # --- NÜMERİK STABİLİTE (NUMERICAL STABILITY) ---
        # Problem: exp(1000) gibi değerler Overflow hatası verir.
        # Çözüm: Tüm değerlerden en büyüğünü çıkararak sayıları küçültmek.
        # Matematiksel olarak olasılık dağılımını değiştirmez.

        # Isı (Temperature) parametresine böl
        scaled_rewards = rewards / self.temperature

        # Max değeri çıkar (Shift)
        stabilized_rewards = scaled_rewards - np.max(scaled_rewards)

        # Exponent al
        exp_values = np.exp(stabilized_rewards)

        # Olasılıkları hesapla
        probabilities = exp_values / np.sum(exp_values)

        # Olasılıklara göre seçim yap
        return np.random.choice(self.n_servers, p=probabilities)

    def update(self, server_id, latency):
        """
        Seçilen sunucunun gerçek yanıt süresine göre Q değerini günceller.
        """
        self.counts[server_id] += 1

        # Learning Rate (Alpha):
        # Sabit bir alpha (örn 0.1) kullanıyoruz çünkü ortam Non-stationary.
        # Ortam değişken olduğu için eski bilgileri yavaşça unutmalıyız.
        alpha = 0.1

        current_estimate = self.q_values[server_id]
        error = latency - current_estimate

        # Yeni Tahmin = Eski Tahmin + Alpha * Hata
        self.q_values[server_id] = current_estimate + alpha * error


# --- 3. SİMÜLASYON VE ANALİZ ---
def main():
    # Ayarlar
    K = 5  # Sunucu sayısı
    N_REQUESTS = 1000  # Toplam istek sayısı
    TAU = 25.0  # Sıcaklık (Temperature)

    servers = [Server(i) for i in range(K)]
    lb = SoftmaxLoadBalancer(K, temperature=TAU)

    latency_history = []
    avg_latency_history = []

    print(f"\nSimülasyon Başlıyor... ({N_REQUESTS} istek)\n")

    for i in range(N_REQUESTS):
        # 1. Softmax ile sunucu seç
        selected_server_id = lb.select_server()

        # 2. İsteği gönder, yanıt süresini al
        latency = servers[selected_server_id].respond()

        # 3. Öğren (Update)
        lb.update(selected_server_id, latency)

        # Kayıt tut
        latency_history.append(latency)
        # Kümülatif ortalama (Grafik için)
        current_avg = np.mean(latency_history)
        avg_latency_history.append(current_avg)

    # --- SONUÇLARI GÖSTER ---
    print("\nSimülasyon Bitti.")
    print("Sunucuların Son Tahmini Latency Değerleri (Q-Values):")
    for i, q in enumerate(lb.q_values):
        print(f"Server {i}: {q:.2f} ms")

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))

    # Ham veriyi (nokta nokta) çizmek yerine hareketli ortalamayı çiziyoruz
    plt.plot(avg_latency_history, label='Softmax Ortalama Latency', color='blue', linewidth=2)

    plt.title(f'Softmax Load Balancer Performans Analizi (Temperature={TAU})')
    plt.xlabel('İstek Sayısı (Zaman)')
    plt.ylabel('Ortalama Gecikme Süresi (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Grafiği kaydet
    plt.savefig('softmax_analysis.png')
    print("\nGrafik 'softmax_analysis.png' dosyasına kaydedildi.")
    plt.show()


if __name__ == "__main__":
    main()