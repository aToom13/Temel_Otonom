# Dursun Web Interface (Frontend)

Bu dizin, React tabanlı SPA (Single Page Application) olarak geliştirilecek yeni web arayüzü kodlarını içerir.

## Başlangıç

1. `npm install` ile bağımlılıkları yükleyin.
2. `npm start` ile geliştirme sunucusunu başlatın.

## Temel Yapı
- `src/` : React kaynak kodları
  - `components/` : Tekil UI bileşenleri (VideoStream, Status, Controls, vb.)
  - `pages/` : Ana sayfa ve alt sayfalar
  - `services/` : API ve WebSocket istemci kodları
  - `App.js` : Uygulama kök bileşeni
  - `index.js` : Giriş noktası

## WebSocket
Gerçek zamanlı veri akışı için Socket.IO istemcisi kullanılacaktır.

## Backend Entegrasyonu
API ve WebSocket endpoint adresleri `.env` dosyası ile yönetilecektir.

---
