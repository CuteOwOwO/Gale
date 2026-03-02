// gamescript.js
let levels = [];
let currentIndex = 0;
let isAnimating = false; 
let currentVideoId = 'taichi'; // 預設

// 計時器變數
let confirmTimer = null;
const CONFIRM_DURATION = 2000; 
const CIRCUMFERENCE = 283;     

const track = document.getElementById('track');
const wifiIcon = document.getElementById('wifi-icon');
const dot = document.querySelector('.dot');

// 🌟 核心改造：非同步初始化系統
async function initSystem() {
    console.log("🛠️ 系統初始化: 讀取 config.json");

    // 從網址列抓取大廳傳來的影片 ID (例如 ?video=towel)
    const urlParams = new URLSearchParams(window.location.search);
    currentVideoId = urlParams.get('video') || 'taichi';

    try {
        // 向伺服器請求 config.json
        const response = await fetch('/static/config.json');
        const configData = await response.json();

        // 3. 檢查影片是否存在，否則退回預設
        if (!configData[currentVideoId]) {
            console.warn(`⚠️ 找不到影片設定: ${currentVideoId}，自動切換為 taichi`);
            currentVideoId = 'taichi';
        }

        const videoConfig = configData[currentVideoId];
        console.log(`🎬 目前載入影片: ${videoConfig.title}，共有 ${videoConfig.key_frames.length} 關`);

        // 根據 JSON 裡的 key_frames 動態生成關卡陣列
        levels = videoConfig.key_frames.map((frame, index) => {
            return {
                id: `${currentVideoId}_${frame}`, 
                frame: frame,                    
                title: `動作訓練 ${index + 1}`,
                image: `${currentVideoId}_${frame}.jpg`, // 預期圖片檔名
                stars: index === 0 ? 3 : 1,    
                locked: index > 2              
            };
        });

        // 5. 資料準備完畢，開始渲染畫面
        renderCards();
        updateLayout();

    } catch (error) {
        console.error("❌ 讀取 config.json 失敗！請確認檔案是否存在。", error);
    }
}

function renderCards() {
    track.innerHTML = '';
    levels.forEach((level, index) => {
        const card = document.createElement('div');
        card.className = `level-card ${level.locked ? 'locked' : ''}`;
        card.id = `card-${index}`;

        let starsHtml = '';
        for(let i=0; i<3; i++) {
            starsHtml += `<span class="star ${i < level.stars ? 'filled' : ''}">★</span>`;
        }

        // 把 loader 加入 HTML
        card.innerHTML = `
            <div class="loader-container">
                <svg class="loader-svg" width="120" height="120" viewBox="0 0 100 100">
                    <circle class="loader-bg" cx="50" cy="50" r="45"></circle>
                    <circle class="loader-progress" cx="50" cy="50" r="45"></circle>
                </svg>
            </div>
            <div class="card-title">Level ${level.id}</div>
            <div class="card-image-wrapper">
               <img src="/static/frames/${level.image}" 
                    alt="${level.title}" 
                    onerror="this.style.display='none'; this.parentNode.innerHTML='<span style=\'font-size:3rem\'>🤸</span>'">
                <div class="lock-overlay">🔒</div>
            </div>
            <div class="card-title" style="font-size: 1.2rem;">${level.title}</div>
            <div class="stars-container">${starsHtml}</div>
        `;
        track.appendChild(card);
    });
    console.log("🃏 卡片渲染完成");
}

function updateLayout() {
    console.log(`🔄 更新佈局: 聚焦卡片 [${currentIndex}]`);
    const cards = document.querySelectorAll('.level-card');
    
    cards.forEach((card, index) => {
        if (index === currentIndex) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });

    const offset = (currentIndex * 320); 
    track.style.transform = `translateX(calc(-${offset}px - 140px))`; 
    
    // 動作完成後，解鎖動畫狀態，並開始轉圈圈
    setTimeout(() => { 
        isAnimating = false; 
        console.log("🛑 滑動動畫結束，準備啟動計時器...");
        startConfirmationTimer(); 
    }, 500);
}

// === 🌟 計時器核心邏輯 ===
function resetConfirmationTimer() {
    if (confirmTimer) {
        clearTimeout(confirmTimer);
        confirmTimer = null;
        console.log("⏸️ 計時器已中斷/重置");
    }
    
    // 保險起見：用 JS 強制把所有圈圈歸零
    document.querySelectorAll('.loader-progress').forEach(progress => {
        progress.style.transition = 'none';
        progress.style.strokeDasharray = CIRCUMFERENCE; 
        progress.style.strokeDashoffset = CIRCUMFERENCE;
        progress.style.stroke = "var(--highlight-color)"; // 恢復橘色
    });
}

function startConfirmationTimer() {
    resetConfirmationTimer(); 
    
    const level = levels[currentIndex];
    
    if (level.locked) {
        console.log(`🔒 關卡 [${currentIndex}] 已上鎖，不啟動計時器`);
        return;
    }

    const activeCard = document.getElementById(`card-${currentIndex}`);
    const progressCircle = activeCard.querySelector('.loader-progress');
    
    if (!progressCircle) {
        console.error("❌ 找不到 loader-progress SVG 元素！");
        return;
    }

    console.log(`⏳ 啟動計時器！關卡 [${currentIndex}] 倒數 3 秒...`);

    // 強制重繪以確保 reset 生效
    progressCircle.getBoundingClientRect(); 

    // 開始畫圈動畫 (給一點微小延遲確保 display:block 已生效)
    setTimeout(() => {
        progressCircle.style.transition = `stroke-dashoffset ${CONFIRM_DURATION}ms linear`;
        progressCircle.style.strokeDashoffset = '0';
    }, 100); 

    // 時間到之後的跳轉動作
    confirmTimer = setTimeout(() => {
        console.log("✅ 時間到！準備跳轉！");
        progressCircle.style.stroke = "#4CAF50"; 
        
        // 拿出記錄好的 frame
        const levelFrame = levels[currentIndex].frame; 
        
        // 帶著兩個參數出發：video=taichi & frame=15
        const targetUrl = `/main_play.html?video=${currentVideoId}&frame=${levelFrame}`;
        console.log(`🚀 執行跳轉：${targetUrl}`);
        
        window.location.href = targetUrl; 
        
    }, CONFIRM_DURATION);
}
// ==============================

function triggerAction(action) {
    if (isAnimating) {
        console.log("⚠️ 正在滑動中，忽略指令:", action);
        return; 
    }

    console.log(`🕹️ 收到指令: ${action}`);

    if (action === 'right') {
        if (currentIndex < levels.length - 1) {
            // 🌟 修正：確定能往右移，才重置計時器
            resetConfirmationTimer(); 
            isAnimating = true;
            currentIndex++;
            updateLayout();
        } else {
            console.log("🧱 已經是最右邊，忽略指令且【不中斷計時】");
        }
    } else if (action === 'left') {
        if (currentIndex > 0) {
            // 🌟 修正：確定能往左移，才重置計時器
            resetConfirmationTimer(); 
            isAnimating = true;
            currentIndex--;
            updateLayout();
        } else {
            console.log("🧱 已經是最左邊，忽略指令且【不中斷計時】");
        }
    } 
   
}

// === WebSocket 連接保留 ===
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
    console.log("🌐 已連線到 Python 後端 (Game 頁面)");
    if (wifiIcon) wifiIcon.style.color = "#4CAF50"; 
    
    if (dot) {
        dot.style.backgroundColor = "#00FF00";
        dot.style.boxShadow = "0 0 15px #00FF00";
    }
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.action === "left") triggerAction('left');
    else if (data.action === "right") triggerAction('right');
};

ws.onclose = () => {
    console.log("🔌 與後端斷開連線");
    if (wifiIcon) wifiIcon.style.color = "red";
};

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') triggerAction('right');
    else if (e.key === 'ArrowLeft') triggerAction('left');
    else if (e.key === 'Enter') triggerAction('enter');
});

window.addEventListener('resize', updateLayout);

// 啟動
initSystem();