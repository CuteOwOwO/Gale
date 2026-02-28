// gamescript.js

const imageFiles = [
    "final_best_poses\\best_pose_frame_15.jpg", 
    "final_best_poses\\best_pose_frame_35.jpg",    
    "final_best_poses\\best_pose_frame_80.jpg",     
    "final_best_poses\\best_pose_frame_101.jpg",   
    "final_best_poses\\best_pose_frame_130.jpg" 
];

let levels = [];
let currentIndex = 0;
// 用來防止手勢或鍵盤連續觸發過快
let isAnimating = false; 

// 取得 HTML 元素
const track = document.getElementById('track');
const wifiIcon = document.getElementById('wifi-icon');

// 1. 初始化
function initLevels() {
    levels = imageFiles.map((imgName, index) => {
        return {
            id: index + 1,
            title: `動作訓練 ${index + 1}`,
            image: imgName, 
            stars: index === 0 ? 3 : 0,    
            locked: index > 2              
        };
    });
    renderCards();
    updateLayout();
}

// 2. 渲染卡片
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

        card.innerHTML = `
            <div class="card-title">Level ${level.id}</div>
            <div class="card-image-wrapper">
                <img src="./static/${level.image}" 
                     alt="${level.title}" 
                     onerror="this.style.display='none'; this.parentNode.innerHTML='<span style=\'font-size:3rem\'>🤸</span>'">
                <div class="lock-overlay">🔒</div>
            </div>
            <div class="card-title" style="font-size: 1.2rem;">${level.title}</div>
            <div class="stars-container">${starsHtml}</div>
        `;
        track.appendChild(card);
    });
}

// 3. 核心佈局 (UI更新)
function updateLayout() {
    const cards = document.querySelectorAll('.level-card');
    const cardWidth = 440; 

    cards.forEach((card, index) => {
        if (index === currentIndex) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });

    // 計算位移
    const offset = (currentIndex * 320); 
    track.style.transform = `translateX(calc(-${offset}px - 140px))`; 
    
    // 動作完成後，解鎖動畫狀態 (0.5s 是配合 CSS transition 的時間)
    setTimeout(() => { isAnimating = false; }, 500);
}

// 4. 動作執行函式 (整合鍵盤與手勢)
function triggerAction(action) {
    if (isAnimating) return; // 如果還在跑動畫，忽略這次指令

    if (action === 'right') {
        if (currentIndex < levels.length - 1) {
            isAnimating = true;
            currentIndex++;
            updateLayout();
        }
    } else if (action === 'left') {
        if (currentIndex > 0) {
            isAnimating = true;
            currentIndex--;
            updateLayout();
        }
    } else if (action === 'enter') {
        const level = levels[currentIndex];
        const card = document.getElementById(`card-${currentIndex}`);
        if (level.locked) {
            // 鎖定時的搖晃特效
            card.style.transform = "translateX(10px) scale(1.1)";
            setTimeout(() => card.style.transform = "translateX(-10px) scale(1.1)", 100);
            setTimeout(() => card.style.transform = "scale(1.1)", 200);
        } else {
            // 進入關卡時的縮放特效
            card.style.transform = "scale(0.9)";
            setTimeout(() => {
                card.style.transform = "scale(1.1)";
                console.log(`準備啟動影像識別：${level.image}`);
                // alert(`進入關卡：${level.image}`); // 測試時可打開這行
            }, 150);
        }
    }
}


const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
    console.log("✅ 已連線到 Python 後端 (Game 頁面)");
    if (wifiIcon) wifiIcon.style.color = "#4CAF50"; // 連線成功讓 WiFi 圖示變綠
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.action === "left") {
        console.log("👈 收到手勢：左");
        triggerAction('left');
    } else if (data.action === "right") {
        console.log("👉 收到手勢：右");
        triggerAction('right');
    }
};

ws.onclose = () => {
    console.log("❌ 與後端斷開連線");
    if (wifiIcon) wifiIcon.style.color = "red";
};

// 6. 鍵盤事件 (保留做測試用)
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') triggerAction('right');
    else if (e.key === 'ArrowLeft') triggerAction('left');
    else if (e.key === 'Enter') triggerAction('enter');
});

// 視窗大小改變時重新計算佈局
window.addEventListener('resize', updateLayout);

// 啟動畫面
initLevels();