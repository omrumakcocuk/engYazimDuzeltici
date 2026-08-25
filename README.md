# Mektup Düzeltici

El yazısıyla yazılmış İngilizce bir mektubun fotoğrafını analiz eder; yanlış kelime veya ifadelerin üzerini çizip doğru hâllerini fotoğrafın üstüne yazar. Sonuç PNG olarak indirilebilir.

## Kurulum

Node.js 20 veya daha yenisi, macOS ve yerel LanguageTool gerekir. Kelime koordinatları macOS ile gelen Apple Vision OCR tarafından ücretsiz ve cihaz üzerinde çıkarılır. LanguageTool kurulumu:

```bash
brew install languagetool
```

Harici npm paketi kullanılmadığı için ayrıca `npm install` çalıştırmak gerekmez.

1. `.env.example` dosyasını `.env` adıyla kopyalayın.
2. `.env` içindeki `OPENAI_API_KEY` değerini kendi OpenAI API anahtarınızla doldurun.
3. Uygulamayı başlatın:

```bash
npm start
```

4. Tarayıcıda `http://localhost:3000` adresini açın.

## Nasıl çalışır?

- Fotoğraf tarayıcıdan sunucuya base64 veri olarak gönderilir.
- Apple Vision OCR görünür kelimeleri, satırları ve gerçek konumlarını cihaz üzerinde çıkarır.
- OpenAI Responses API her satırın tam düzeltilmiş cümlesini üretir; modelden koordinat veya parça düzeltmesi istenmez.
- Yerel LanguageTool düzeltilmiş cümleyi kontrol eder ve uygun kural düzeltmelerini uygular.
- Kod, OCR metni ile kabul edilen son cümle arasında token diff yaparak `insert`, `replace` ve `delete` işlemlerini çıkarır.
- Tarayıcıdaki Canvas API, hatalı kısmın üzerine mavi çizgi ve doğrusunu el yazısı stiliyle ekler.
- API anahtarı hiçbir zaman tarayıcıya gönderilmez.

Model koordinatları görselden tahmin ettiği için özellikle eğik, bulanık veya gölgeli fotoğraflarda yerleşim birkaç piksel sapabilir. En iyi sonuç için fotoğrafı kâğıda dik açıyla ve iyi ışıkta çekin.
