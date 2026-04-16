// Web Worker pour traiter le WebSocket en arrière-plan
// Les navigateurs ne brident PAS les Web Workers quand l'onglet devient inactif.

const APP_ID = 1089; 
const wsUrl = `wss://ws.binaryws.com/websockets/v3?app_id=${APP_ID}`;
let ws = null;
let pingInterval = null;

// Connecte et garde en vie
function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    // Demande du stream
    ws.send(JSON.stringify({
      ticks: "BOOM1000",
      subscribe: 1
    }));
    
    // Ping anti-timeout Deriv (toutes les 25s)
    if (pingInterval) clearInterval(pingInterval);
    pingInterval = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ ping: 1 }));
      }
    }, 25000);
  };

  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.msg_type === "tick" && data.tick) {
      const newTick = {
        timestamp: data.tick.epoch,
        price: data.tick.quote,
      };
      // Envoi du tick pré-traité au thread principal (UI)
      postMessage({ type: 'TICK', payload: newTick });
    }
  };

  ws.onclose = () => {
    postMessage({ type: 'STATUS', payload: 'DISCONNECTED' });
    if (pingInterval) clearInterval(pingInterval);
    // Reconnexion auto
    setTimeout(connect, 3000);
  };

  ws.onerror = (err) => {
    ws.close();
  };
}

// Ecoute les ordres du thread principal (Dashboard)
self.onmessage = (e) => {
  if (e.data.type === 'START') {
    connect();
  } else if (e.data.type === 'STOP') {
    if (pingInterval) clearInterval(pingInterval);
    if (ws) ws.close();
  }
};
