import json
import websocket
import threading

class BinanceStream:
    def __init__(self, symbol='ethusdt', interval='5m'):
        self.symbol = symbol.lower()
        self.interval = interval
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        self.latest_kline = None
        self.ws = None

    def on_message(self, ws, message):
        data = json.loads(message)
        kline = data['k']
        if kline['x']: # Candle closed
            self.latest_kline = kline
            print(f"New closed candle: {kline['c']}")

    def on_error(self, ws, error):
        print(f"WS Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WS Closed")

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

if __name__ == "__main__":
    stream = BinanceStream()
    stream.start()
    import time
    while True:
        time.sleep(1)
