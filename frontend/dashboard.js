const video = document.getElementById("video")
const uploadInput = document.getElementById("upload")
const pickFileBtn = document.getElementById("pickFileBtn")
const selectedFileText = document.getElementById("selectedFileText")
const uploadPreview = document.getElementById("uploadPreview")
const uploadProgressBar = document.getElementById("uploadProgressBar")
const uploadProgressText = document.getElementById("uploadProgressText")
const statusText = document.getElementById("statusText")
const textInput = document.getElementById("textInput")
const textStatus = document.getElementById("textStatus")
const cameraState = document.getElementById("cameraState")
const uploadState = document.getElementById("uploadState")
const liveMeta = document.getElementById("liveMeta")
const liveAudioMeta = document.getElementById("liveAudioMeta")
const liveStatusText = document.getElementById("liveStatusText")
const resultState = document.getElementById("resultState")
const statTotal = document.getElementById("statTotal")
const statDominant = document.getElementById("statDominant")
const statRatio = document.getElementById("statRatio")
const statAudio = document.getElementById("statAudio")
const statFusion = document.getElementById("statFusion")
const statText = document.getElementById("statText")
const modeGate = document.getElementById("modeGate")
const workspaceGrid = document.getElementById("workspaceGrid")
const modeToolbar = document.getElementById("modeToolbar")
const currentModeText = document.getElementById("currentModeText")
const realtimePanel = document.getElementById("realtimePanel")
const realtimeChartPanel = document.getElementById("realtimeChartPanel")
const uploadPanel = document.getElementById("uploadPanel")
const uploadResultPanel = document.getElementById("uploadResultPanel")
const faceOverlay = document.getElementById("faceOverlay")
const faceBox = document.getElementById("faceBox")
const faceBoxLabel = document.getElementById("faceBoxLabel")
const floatingCompanion = document.getElementById("floatingCompanion")
const companionDragHandle = document.getElementById("companionDragHandle")
const companionAvatar = document.getElementById("companionAvatar")
const companionEmotion = document.getElementById("companionEmotion")
const companionMessage = document.getElementById("companionMessage")
const companionDialog = document.getElementById("companionDialog")
const langZhBtn = document.getElementById("langZhBtn")
const langEnBtn = document.getElementById("langEnBtn")

let stream = null
let chart = null
let distributionChart = null
let uploadTrendChart = null
let timeline = []
let uploadTimeline = []
let uploadAudioEmotion = null
let uploadFusionEmotion = null
let uploadFusionConfidence = 0
let uploadTextEmotion = null
let captureTimer = null
let micStream = null
let mediaRecorder = null
let micChunks = []
let micCycleTimer = null
let micLoopEnabled = false
let currentMode = null
let lastFaceBox = null
let lastFaceEmotion = null
let lastRealtimeVideoEmotion = null
let lastRealtimeVideoConfidence = 0
let lastRealtimeAudioEmotion = null
let lastRealtimeAudioConfidence = 0
let lastRealtimeAudioError = null
const EMOTIONS = ["angry", "sad", "neutral", "happy"]
const LIVE_SMOOTH_WINDOW = 5
const liveVoteBuffer = []
const companionHistory = []
const COMPANION_POSITION_KEY = "emotionCompanionPositionV1"
const UI_LANG_KEY = "emotionUiLang"
const COMPANION_MOOD_CLASS = {
angry: "mood-angry",
sad: "mood-sad",
neutral: "mood-neutral",
happy: "mood-happy"
}

const I18N = {
zh: {
title: "多模态情绪分析系统",
"hero.tag": "Emotion Lab",
"hero.title": "多模态情绪分析",
"hero.sub": "上传视频或使用实时摄像头流，检测面部情绪变化趋势。",
"toolbar.realtime": "实时模式",
"toolbar.upload": "上传模式",
"toolbar.back": "返回",
"gate.title": "选择你的分析流程",
"gate.sub": "先选择一种模式开始，之后可在顶部工具栏随时切换。",
"gate.enterRealtime": "进入实时模式",
"gate.enterUpload": "进入上传模式",
"panel.liveCamera": "实时摄像头",
"panel.videoUpload": "视频上传",
"panel.visualTimeline": "视觉情绪时间线",
"panel.realtimeFusion": "实时音频融合",
"panel.uploadInsights": "上传分析结果",
"upload.selectFile": "选择视频文件",
"upload.pickFile": "选择文件",
"upload.noFile": "未选择文件",
"text.label": "文本输入（单独分析）",
"text.placeholder": "来和小Q聊聊天吧！",
"action.startCamera": "启动摄像头",
"action.stopCamera": "停止摄像头",
"action.analyzeText": "分析文本",
"action.analyzeUpload": "分析上传视频",
"action.exportJson": "导出 JSON",
"action.exportCsv": "导出 CSV",
"stats.total": "时间线点数",
"stats.dominant": "主导情绪",
"stats.ratio": "主导占比",
"stats.audio": "音频情绪",
"stats.fusion": "融合情绪",
"stats.fusionScore": "融合置信度",
"stats.text": "文本情绪",
"chart.trend": "情绪随时间变化",
"chart.distribution": "情绪分布",
"companion.title": "情绪小宠物",
"companion.drag": "拖动",
"companion.dragAria": "拖动宠物",
"toolbar.modeUnknown": "模式：--",
"toolbar.modeRealtime": "模式：实时",
"toolbar.modeUpload": "模式：上传",
"status.switchRealtime": "已切换到实时模式。",
"status.switchUpload": "已切换到上传模式。",
"status.cameraStarted": "摄像头已启动。",
"status.cameraDenied": "无法访问摄像头权限。",
"status.liveRunning": "实时分析运行中。",
"status.liveDenied": "摄像头访问被拒绝。",
"status.liveIdle": "摄像头空闲中。",
"status.liveNoFace": "当前帧未检测到人脸。",
"status.liveOk": "实时分析正常。",
"status.liveRequestFail": "实时请求失败：{message}",
"status.micUnsupported": "浏览器不支持 MediaRecorder，实时音频已关闭。",
"status.micInitFail": "麦克风初始化失败：{message}",
"status.micStartFail": "麦克风启动失败：{message}",
"status.micDenied": "麦克风权限被拒绝，实时音频已关闭。",
"status.micCycleFail": "麦克风循环录制失败：{message}",
"status.audioChunkFail": "实时音频失败：{message}",
"status.chooseFileFirst": "请先选择视频文件。",
"status.uploading": "视频上传中...",
"status.processing": "上传完成，后端正在分析帧...",
"status.parseFail": "上传成功，但解析响应失败。",
"status.videoDoneWithAudioFail": "视频分析完成（{count} 个点），但音频失败：{error}",
"status.videoDone": "分析完成：{count} 个时间线点。",
"status.uploadFail": "上传失败（{code}），请检查后端日志。",
"status.networkFail": "网络错误，后端可能未启动。",
"status.inputTextFirst": "请先输入文本。",
"status.textAnalyzing": "文本分析中...",
"status.textResult": "文本情绪：{emotion}（置信度：{confidence}）",
"status.textFail": "文本分析失败：{message}",
"status.waitingFile": "等待选择文件。",
"status.waitingText": "等待文本输入。",
"status.selectedFile": "已选择：{name}",
"status.noUploadToExport": "暂无可导出的上传分析结果。",
"status.exportJsonOk": "JSON 报告已导出。",
"status.exportCsvOk": "CSV 报告已导出。",
"live.meta": "延迟：{latency} ms | 最近：{last} | 平滑：{smooth}",
"live.metaNoFace": "延迟：-- ms | 最近：无人脸 | 平滑：--",
"live.metaIdle": "延迟：-- ms | 最近情绪：--",
"live.audioMeta": "音频：{audio} | 融合：{fusion}",
"audio.unsupported": "不支持",
"audio.videoOnly": "仅视频",
"audio.recorderFailed": "录制器失败",
"audio.startFailed": "启动失败",
"audio.listening": "监听中",
"audio.unavailable": "不可用",
"audio.error": "错误",
"face.primary": "主目标人脸",
"face.primaryWithEmotion": "主目标人脸 · {emotion}",
"state.idle": "空闲",
"state.live": "直播中",
"state.ready": "就绪",
"state.uploading": "上传中",
"state.processing": "处理中",
"state.done": "完成",
"state.error": "错误",
"state.noData": "暂无数据",
"state.readyResult": "就绪",
"state.analyzed": "已分析",
"chart.emotionLabel": "情绪",
"chart.count": "数量",
"chart.time": "时间（秒）",
"emotion.angry": "愤怒",
"emotion.sad": "难过",
"emotion.neutral": "平静",
"emotion.happy": "开心",
"emotion.unknown": "未知",
"companion.neutralPrompt": "分析后我会给你安慰、鼓励和建议。",
"companion.idleDialog": "小Q：我在顶部陪着你。",
"companion.idleDialog2": "小Q：上传或实时分析后，我都会陪你说句话。",
"companion.emotion.happy": "当前情绪：开心",
"companion.emotion.neutral": "当前情绪：平静",
"companion.emotion.sad": "当前情绪：低落",
"companion.emotion.angry": "当前情绪：烦躁",
"companion.msg.happy": "你现在状态很棒，继续保持这个节奏。给自己一个小奖励，把这份好心情延续下去。",
"companion.msg.neutral": "你的情绪比较稳定，适合专注做事。可以安排一个小目标，稳稳推进就很好。",
"companion.msg.sad": "辛苦了，你已经很努力了。先慢慢呼吸几次，给自己一点时间，我会一直陪着你。",
"companion.msg.angry": "我理解你现在不舒服。先停一下，喝口水或走动两分钟，等身体放松后再处理问题。",
"companion.dialog.happy": ["小Q：今天这个能量感，真赞。", "小Q：你在发光，继续冲。", "小Q：这份开心值得收藏。"],
"companion.dialog.neutral": ["小Q：稳稳的状态也很珍贵。", "小Q：我们一步一步来。", "小Q：保持节奏，你已经很好了。"],
"companion.dialog.sad": ["小Q：别急，我陪你缓一缓。", "小Q：先照顾好自己，事情慢慢来。", "小Q：你不需要一个人扛着。"],
"companion.dialog.angry": ["小Q：先暂停一下，呼吸四次。", "小Q：把火气放下，我们再出发。", "小Q：你可以慢一点，但不要否定自己。"],
"companion.trend.upHappy": "小Q：我看到你状态在上扬，继续保持。",
"companion.trend.up": "小Q：你的情绪在慢慢回暖，做得很好。",
"companion.trend.down": "小Q：我注意到你有点累了，先休息一分钟。",
"companion.defaultDialog": "小Q：我在这里。"
},
en: {
title: "Multimodal Emotion Analysis System",
"hero.tag": "Emotion Lab",
"hero.title": "Multimodal Emotion Analysis",
"hero.sub": "Upload a video or use live camera stream to detect facial emotion trends in real time.",
"toolbar.realtime": "Realtime Mode",
"toolbar.upload": "Upload Mode",
"toolbar.back": "Back",
"gate.title": "Select Your Workflow",
"gate.sub": "Choose one mode to start. You can switch anytime from the top toolbar.",
"gate.enterRealtime": "Enter Realtime Mode",
"gate.enterUpload": "Enter Upload Mode",
"panel.liveCamera": "Live Camera",
"panel.videoUpload": "Video Upload",
"panel.visualTimeline": "Visual Emotion Timeline",
"panel.realtimeFusion": "Realtime Audio Fusion",
"panel.uploadInsights": "Upload Insights",
"upload.selectFile": "Select Video File",
"upload.pickFile": "Choose File",
"upload.noFile": "No file selected",
"text.label": "Text Input (Standalone)",
"text.placeholder": "Have a quick chat with Q!",
"action.startCamera": "Start Camera",
"action.stopCamera": "Stop Camera",
"action.analyzeText": "Analyze Text",
"action.analyzeUpload": "Analyze Uploaded Video",
"action.exportJson": "Export JSON",
"action.exportCsv": "Export CSV",
"stats.total": "Timeline Points",
"stats.dominant": "Dominant Emotion",
"stats.ratio": "Dominant Ratio",
"stats.audio": "Audio Emotion",
"stats.fusion": "Fused Emotion",
"stats.fusionScore": "Fusion Confidence",
"stats.text": "Text Emotion",
"chart.trend": "Emotion Trend Over Time",
"chart.distribution": "Emotion Distribution",
"companion.title": "Emo Pet",
"companion.drag": "Drag",
"companion.dragAria": "Drag companion",
"toolbar.modeUnknown": "Mode: --",
"toolbar.modeRealtime": "Mode: Realtime",
"toolbar.modeUpload": "Mode: Upload",
"status.switchRealtime": "Switched to realtime mode.",
"status.switchUpload": "Switched to upload mode.",
"status.cameraStarted": "Camera started.",
"status.cameraDenied": "Unable to access camera permissions.",
"status.liveRunning": "Realtime analysis running.",
"status.liveDenied": "Camera access denied.",
"status.liveIdle": "Camera is idle.",
"status.liveNoFace": "No face detected in current frame.",
"status.liveOk": "Realtime analysis ok.",
"status.liveRequestFail": "Realtime request failed: {message}",
"status.micUnsupported": "Browser does not support MediaRecorder. Realtime audio disabled.",
"status.micInitFail": "Mic recorder init failed: {message}",
"status.micStartFail": "Mic recorder start failed: {message}",
"status.micDenied": "Mic access denied. Realtime audio disabled.",
"status.micCycleFail": "Mic recorder cycle failed: {message}",
"status.audioChunkFail": "Realtime audio failed: {message}",
"status.chooseFileFirst": "Please choose a video file first.",
"status.uploading": "Uploading video...",
"status.processing": "Upload complete. Backend is analyzing frames...",
"status.parseFail": "Upload succeeded, but failed to parse response.",
"status.videoDoneWithAudioFail": "Video done ({count} points), but audio failed: {error}",
"status.videoDone": "Analysis complete: {count} timeline points.",
"status.uploadFail": "Upload failed ({code}). Please check backend logs.",
"status.networkFail": "Network error. Backend may not be running.",
"status.inputTextFirst": "Please input text first.",
"status.textAnalyzing": "Analyzing text...",
"status.textResult": "Text emotion: {emotion} (confidence: {confidence})",
"status.textFail": "Text analysis failed: {message}",
"status.waitingFile": "Waiting for file selection.",
"status.waitingText": "Waiting for text input.",
"status.selectedFile": "Selected: {name}",
"status.noUploadToExport": "No upload analysis available to export.",
"status.exportJsonOk": "JSON report exported.",
"status.exportCsvOk": "CSV report exported.",
"live.meta": "Latency: {latency} ms | Last: {last} | Smoothed: {smooth}",
"live.metaNoFace": "Latency: -- ms | Last: no-face | Smoothed: --",
"live.metaIdle": "Latency: -- ms | Last emotion: --",
"live.audioMeta": "Audio: {audio} | Fusion: {fusion}",
"audio.unsupported": "unsupported",
"audio.videoOnly": "video-only",
"audio.recorderFailed": "recorder-failed",
"audio.startFailed": "start-failed",
"audio.listening": "listening",
"audio.unavailable": "unavailable",
"audio.error": "error",
"face.primary": "Primary Face",
"face.primaryWithEmotion": "Primary Face · {emotion}",
"state.idle": "Idle",
"state.live": "Live",
"state.ready": "Ready",
"state.uploading": "Uploading",
"state.processing": "Processing",
"state.done": "Done",
"state.error": "Error",
"state.noData": "No Data",
"state.readyResult": "Ready",
"state.analyzed": "Analyzed",
"chart.emotionLabel": "Emotion",
"chart.count": "Count",
"chart.time": "Time (s)",
"emotion.angry": "Angry",
"emotion.sad": "Sad",
"emotion.neutral": "Neutral",
"emotion.happy": "Happy",
"emotion.unknown": "Unknown",
"companion.neutralPrompt": "After analysis, I will give you support, encouragement, and suggestions.",
"companion.idleDialog": "Q: I'm here with you at the top.",
"companion.idleDialog2": "Q: After upload or realtime analysis, I'll always say something for you.",
"companion.emotion.happy": "Current mood: Happy",
"companion.emotion.neutral": "Current mood: Calm",
"companion.emotion.sad": "Current mood: Down",
"companion.emotion.angry": "Current mood: Irritated",
"companion.msg.happy": "You are in a great state right now. Keep this rhythm and reward yourself a little.",
"companion.msg.neutral": "Your emotion is stable now, which is great for focus. Set a small goal and move steadily.",
"companion.msg.sad": "You have tried hard. Take a few slow breaths and give yourself a little time. I am with you.",
"companion.msg.angry": "I understand this feels uncomfortable. Pause for a moment, drink some water, then continue when relaxed.",
"companion.dialog.happy": ["Q: Your energy today is amazing.", "Q: You're shining, keep going.", "Q: This joy is worth saving."],
"companion.dialog.neutral": ["Q: A stable state is precious too.", "Q: Let's go step by step.", "Q: Keep your pace, you're doing well."],
"companion.dialog.sad": ["Q: It's okay, let's slow down together.", "Q: Take care of yourself first.", "Q: You don't have to carry it alone."],
"companion.dialog.angry": ["Q: Let's pause and take four breaths.", "Q: Put down the anger and restart.", "Q: Slow is okay, don't deny yourself."],
"companion.trend.upHappy": "Q: I can see your state rising, keep it up.",
"companion.trend.up": "Q: Your mood is warming up, well done.",
"companion.trend.down": "Q: You seem tired now. Take a short break.",
"companion.defaultDialog": "Q: I'm here."
}
}

const API_BASE = "http://127.0.0.1:5000"

let companionDragOffsetX = 0
let companionDragOffsetY = 0
let isCompanionDragging = false
let currentLang = loadUiLanguage()

uploadInput.addEventListener("change", handleFilePreview)
if(pickFileBtn){
pickFileBtn.addEventListener("click", ()=>uploadInput.click())
}
textInput.addEventListener("keydown", onTextInputEnterAnalyze)
document.addEventListener("keydown", onGlobalEnterAnalyze)
window.addEventListener("resize", redrawFaceOverlay)
initCompanionDrag()
initLanguageSwitcher()
applyLanguage(currentLang)


function t(key, vars = {}){
const table = I18N[currentLang] || I18N.zh
let template = table[key]
if(template === undefined){
template = I18N.zh[key] !== undefined ? I18N.zh[key] : key
}
if(typeof template !== "string"){
return template
}
return template.replace(/\{(\w+)\}/g, (_, name)=> String(vars[name] ?? ""))
}


function initLanguageSwitcher(){
if(langZhBtn){
langZhBtn.addEventListener("click", ()=>applyLanguage("zh"))
}
if(langEnBtn){
langEnBtn.addEventListener("click", ()=>applyLanguage("en"))
}
}


function applyLanguage(lang){
currentLang = lang === "en" ? "en" : "zh"
document.documentElement.lang = currentLang === "en" ? "en" : "zh-CN"
saveUiLanguage(currentLang)

document.querySelectorAll("[data-i18n]").forEach((el)=>{
const key = el.getAttribute("data-i18n")
if(key){
el.textContent = t(key)
}
})
document.querySelectorAll("[data-i18n-placeholder]").forEach((el)=>{
const key = el.getAttribute("data-i18n-placeholder")
if(key){
el.setAttribute("placeholder", t(key))
}
})
document.querySelectorAll("[data-i18n-aria]").forEach((el)=>{
const key = el.getAttribute("data-i18n-aria")
if(key){
el.setAttribute("aria-label", t(key))
}
})

document.title = t("title")
if(langZhBtn && langEnBtn){
langZhBtn.classList.toggle("active", currentLang === "zh")
langEnBtn.classList.toggle("active", currentLang === "en")
}

refreshLocalizedUI()
}


function refreshLocalizedUI(){
if(currentMode === "realtime"){
currentModeText.textContent = t("toolbar.modeRealtime")
}else if(currentMode === "upload"){
currentModeText.textContent = t("toolbar.modeUpload")
}else{
currentModeText.textContent = t("toolbar.modeUnknown")
}

if(!stream){
cameraState.textContent = t("state.idle")
if(liveStatusText && !liveStatusText.classList.contains("ok") && !liveStatusText.classList.contains("error")){
liveStatusText.textContent = t("status.liveIdle")
}
}

if(!uploadTimeline.length){
resultState.textContent = t("state.noData")
}

if(uploadState){
const current = (uploadState.textContent || "").trim()
if(
current === "就绪" || current === "Ready" ||
current === "空闲" || current === "Idle"
){
uploadState.textContent = t("state.ready")
}
}

if(selectedFileText){
const hasFile = uploadInput && uploadInput.files && uploadInput.files.length > 0
selectedFileText.textContent = hasFile ? uploadInput.files[0].name : t("upload.noFile")
}

if(textStatus && !textStatus.classList.contains("ok") && !textStatus.classList.contains("error")){
textStatus.textContent = t("status.waitingText")
}

if(statusText && !statusText.classList.contains("ok") && !statusText.classList.contains("error") && (!uploadInput.files || uploadInput.files.length === 0)){
statusText.textContent = t("status.waitingFile")
}

if(!timeline.length && !stream){
liveMeta.textContent = t("live.metaIdle")
liveAudioMeta.textContent = t("live.audioMeta", {audio: "--", fusion: "--"})
}

if(uploadTimeline.length){
drawDistribution(uploadTimeline)
drawUploadTrend(uploadTimeline)
}

if(companionHistory.length === 0){
resetCompanion(t("companion.neutralPrompt"), t("companion.idleDialog"))
}
}


function saveUiLanguage(lang){
try{
window.localStorage.setItem(UI_LANG_KEY, lang)
}catch(_e){
}
}


function loadUiLanguage(){
try{
const v = window.localStorage.getItem(UI_LANG_KEY)
return v === "en" ? "en" : "zh"
}catch(_e){
return "zh"
}
}


function emotionText(emotion){
if(!emotion || emotion === "--"){
return "--"
}
if(!EMOTIONS.includes(emotion) && emotion !== "unknown"){
return emotion
}
return t(`emotion.${emotion}`)
}


function onTextInputEnterAnalyze(event){
if(event.key !== "Enter"){
return
}

if(event.shiftKey){
return
}

event.preventDefault()
analyzeTextInput()
}


function onGlobalEnterAnalyze(event){
if(event.key !== "Enter"){
return
}

if(event.shiftKey || event.ctrlKey || event.altKey || event.metaKey || event.isComposing){
return
}

const target = event.target
const tag = target && target.tagName ? String(target.tagName).toLowerCase() : ""

// Ignore native button activation and text area handling in its own listener.
if(tag === "button" || tag === "textarea"){
return
}

if(tag === "input" && target && target.type !== "file"){
return
}

if(currentMode !== "upload"){
return
}

if(uploadInput && uploadInput.files && uploadInput.files.length > 0){
event.preventDefault()
uploadVideo()
}
}


function enterMode(mode){
modeGate.classList.add("hidden")
workspaceGrid.classList.remove("hidden")
modeToolbar.classList.remove("hidden")
switchMode(mode)
}


function switchMode(mode){
currentMode = mode
currentModeText.textContent = mode === "realtime" ? t("toolbar.modeRealtime") : t("toolbar.modeUpload")

workspaceGrid.classList.remove("mode-realtime", "mode-upload")

if(mode === "realtime"){
workspaceGrid.classList.add("mode-realtime")
realtimePanel.classList.remove("hidden")
realtimeChartPanel.classList.remove("hidden")
uploadPanel.classList.add("hidden")
uploadResultPanel.classList.add("hidden")
setStatus(t("status.switchRealtime"), "")
}else{
workspaceGrid.classList.add("mode-upload")
stopCamera()
realtimePanel.classList.add("hidden")
realtimeChartPanel.classList.add("hidden")
uploadPanel.classList.remove("hidden")
uploadResultPanel.classList.remove("hidden")
setStatus(t("status.switchUpload"), "")
}
}


function backToModeSelection(){
stopCamera()
workspaceGrid.classList.add("hidden")
workspaceGrid.classList.remove("mode-realtime", "mode-upload")
modeToolbar.classList.add("hidden")
modeGate.classList.remove("hidden")
currentMode = null
currentModeText.textContent = t("toolbar.modeUnknown")
}


function startCamera(){

if(captureTimer){
clearInterval(captureTimer)
}

navigator.mediaDevices.getUserMedia({video:true, audio:true})
.then(s=>{

stream = s

video.srcObject = s

cameraState.textContent = t("state.live")

captureTimer = setInterval(captureFrame,1000)

setStatus(t("status.cameraStarted"),"ok")
setLiveStatus(t("status.liveRunning"), "ok")
startMicrophoneCapture(s)

})
.catch(()=>{
setStatus(t("status.cameraDenied"),"error")
setLiveStatus(t("status.liveDenied"), "error")
})

}


function stopCamera(){

micLoopEnabled = false
if(micCycleTimer){
clearTimeout(micCycleTimer)
micCycleTimer = null
}

if(mediaRecorder && mediaRecorder.state !== "inactive"){
mediaRecorder.stop()
}

if(micStream){
micStream.getTracks().forEach(track=>track.stop())
micStream = null
}

mediaRecorder = null
lastRealtimeVideoEmotion = null
lastRealtimeAudioEmotion = null
lastRealtimeAudioError = null
resetRealtimeCompanion()

if(stream){

stream.getTracks().forEach(track=>track.stop())

stream = null

}

if(captureTimer){
clearInterval(captureTimer)
captureTimer = null
}

cameraState.textContent = t("state.idle")
liveMeta.textContent = t("live.metaIdle")
liveAudioMeta.textContent = t("live.audioMeta", {audio: "--", fusion: "--"})
setLiveStatus(t("status.liveIdle"), "")
clearFaceOverlay()

}


function captureFrame(){

if(!video.videoWidth || !video.videoHeight){
return
}

const canvas = document.createElement("canvas")

canvas.width = video.videoWidth
canvas.height = video.videoHeight

const ctx = canvas.getContext("2d")

ctx.drawImage(video,0,0)

const data = canvas.toDataURL("image/jpeg")
const startedAt = performance.now()

fetch(`${API_BASE}/analyze_frame`,{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({image:data})

})

.then(r=>{
if(!r.ok){
throw new Error(`HTTP ${r.status}`)
}
return r.json()
})
.then(d=>{

if(d.faceDetected === false){
liveMeta.textContent = t("live.metaNoFace")
lastRealtimeVideoEmotion = null
updateLiveAudioMeta()
setLiveStatus(t("status.liveNoFace"), "error")
clearFaceOverlay()
return
}

const smoothedEmotion = getSmoothedEmotion(d.emotion)
lastRealtimeVideoEmotion = smoothedEmotion
lastRealtimeVideoConfidence = Number(d.confidence || 0)
updateChart(smoothedEmotion)
updateRealtimeCompanion(smoothedEmotion)
const latency = Math.round(performance.now() - startedAt)
liveMeta.textContent = t("live.meta", {
latency,
last: emotionText(d.emotion || "unknown"),
smooth: emotionText(smoothedEmotion)
})
updateLiveAudioMeta()
setLiveStatus(t("status.liveOk"), "ok")
drawFaceOverlay(d.faceBox, smoothedEmotion)

})
.catch((err)=>{
setLiveStatus(t("status.liveRequestFail", {message: err.message}), "error")
clearFaceOverlay()
})

}


function startMicrophoneCapture(mediaStream){

const beginRecorder = (audioStream)=>{
micStream = audioStream
liveAudioMeta.textContent = t("live.audioMeta", {audio: "...", fusion: "--"})

if(typeof MediaRecorder === "undefined"){
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.unsupported"), fusion: t("audio.videoOnly")})
setLiveStatus(t("status.micUnsupported"), "error")
return
}

const preferredType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
? "audio/webm;codecs=opus"
: "audio/webm"

try{
mediaRecorder = MediaRecorder.isTypeSupported(preferredType)
? new MediaRecorder(micStream, {mimeType: preferredType})
: new MediaRecorder(micStream)
}catch(err){
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.recorderFailed"), fusion: t("audio.videoOnly")})
setLiveStatus(t("status.micInitFail", {message: err.message}), "error")
return
}

mediaRecorder.ondataavailable = (event)=>{
if(event.data && event.data.size > 0){
micChunks.push(event.data)
}
}

mediaRecorder.onstop = ()=>{
if(micChunks.length > 0){
const blob = new Blob(micChunks, {type: mediaRecorder.mimeType || "audio/webm"})
micChunks = []
uploadAudioChunk(blob)
}

if(micLoopEnabled && mediaRecorder && mediaRecorder.state === "inactive"){
startMicCycle()
}
}

try{
micLoopEnabled = true
startMicCycle()
}catch(err){
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.startFailed"), fusion: t("audio.videoOnly")})
setLiveStatus(t("status.micStartFail", {message: err.message}), "error")
return
}
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.listening"), fusion: "--"})
lastRealtimeAudioError = null
}

if(mediaStream && mediaStream.getAudioTracks().length > 0){
const audioOnlyStream = new MediaStream(mediaStream.getAudioTracks())
beginRecorder(audioOnlyStream)
return
}

navigator.mediaDevices.getUserMedia({audio:true})
.then(s=>{
beginRecorder(s)
})
.catch(()=>{
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.unavailable"), fusion: t("audio.videoOnly")})
setLiveStatus(t("status.micDenied"), "error")
})

}


function startMicCycle(){
if(!mediaRecorder || !micLoopEnabled){
return
}

micChunks = []
try{
mediaRecorder.start()
}catch(err){
liveAudioMeta.textContent = t("live.audioMeta", {audio: t("audio.startFailed"), fusion: t("audio.videoOnly")})
setLiveStatus(t("status.micCycleFail", {message: err.message}), "error")
return
}

if(micCycleTimer){
clearTimeout(micCycleTimer)
}
micCycleTimer = setTimeout(()=>{
if(mediaRecorder && mediaRecorder.state === "recording"){
mediaRecorder.stop()
}
}, 2500)
}


function uploadAudioChunk(blob){
const form = new FormData()
let ext = "webm"
if(mediaRecorder && typeof mediaRecorder.mimeType === "string"){
if(mediaRecorder.mimeType.includes("ogg")){
ext = "ogg"
}else if(mediaRecorder.mimeType.includes("mp4")){
ext = "mp4"
}else if(mediaRecorder.mimeType.includes("wav")){
ext = "wav"
}
}
form.append("audio", blob, `chunk_${Date.now()}.${ext}`)

fetch(`${API_BASE}/analyze_audio_chunk`, {
method: "POST",
body: form
})
.then(async (r)=>{
if(!r.ok){
let detail = ""
try{
const payload = await r.json()
detail = payload && payload.error ? `: ${payload.error}` : ""
}catch(_e){
detail = ""
}
throw new Error(`HTTP ${r.status}${detail}`)
}
return r.json()
})
.then(data=>{
lastRealtimeAudioEmotion = data.audioEmotion || null
lastRealtimeAudioConfidence = Number(data.confidence || 0)
lastRealtimeAudioError = null
updateLiveAudioMeta()
})
.catch((err)=>{
lastRealtimeAudioEmotion = null
lastRealtimeAudioConfidence = 0
lastRealtimeAudioError = err.message
updateLiveAudioMeta()
setLiveStatus(t("status.audioChunkFail", {message: err.message}), "error")
})
}


function getRealtimeFusionResult(videoEmotion, videoConf, audioEmotion, audioConf){
if(!videoEmotion && !audioEmotion){
return {emotion: "--", confidence: 0}
}

if(videoEmotion && !audioEmotion){
return {emotion: videoEmotion, confidence: videoConf || 0}
}

if(audioEmotion && !videoEmotion){
return {emotion: audioEmotion, confidence: audioConf || 0}
}

const votes = {
angry: 0,
sad: 0,
neutral: 0,
happy: 0,
}

votes[videoEmotion] = (votes[videoEmotion] || 0) + 0.6 * Math.max(0, Math.min(1, Number(videoConf || 0.5)))
votes[audioEmotion] = (votes[audioEmotion] || 0) + 0.4 * Math.max(0, Math.min(1, Number(audioConf || 0.5)))

let bestEmotion = videoEmotion
let bestScore = -1
Object.keys(votes).forEach((k)=>{
if(votes[k] > bestScore){
bestScore = votes[k]
bestEmotion = k
}
})

const denom = votes.angry + votes.sad + votes.neutral + votes.happy
return {
emotion: bestEmotion,
confidence: denom > 0 ? (bestScore / denom) : 0,
}
}


function updateLiveAudioMeta(){
const audioText = lastRealtimeAudioError ? t("audio.error") : emotionText(lastRealtimeAudioEmotion || "unknown")
const fusion = getRealtimeFusionResult(
lastRealtimeVideoEmotion,
lastRealtimeVideoConfidence,
lastRealtimeAudioEmotion,
lastRealtimeAudioConfidence,
)
const fusionText = emotionText(fusion.emotion)
const fusionPretty = fusionText === t("emotion.unknown") || fusionText === "--"
? "--"
: `${fusionText} (${Math.round((fusion.confidence || 0) * 100)}%)`
liveAudioMeta.textContent = t("live.audioMeta", {
audio: audioText === t("emotion.unknown") ? "--" : audioText,
fusion: fusionPretty,
})
}


function updateChart(emotion){

if(!EMOTIONS.includes(emotion)){
return
}

timeline.push(emotion)

if(timeline.length>20){

timeline.shift()

}

const map={
angry:0,
sad:1,
neutral:2,
happy:3
}

const values=timeline.map(e=>map[e]).filter(v=>v !== undefined)

if(chart==null){

chart = new Chart(

document.getElementById("visualChart"),

{

type:"line",

data:{

labels:timeline.map((_,i)=>i),

datasets:[{

label:t("chart.emotionLabel"),

data:values,
borderColor:"#1f6f78",
backgroundColor:"rgba(31, 111, 120, 0.22)",
tension:0.36,
fill:true

}]

},

options:{
responsive:true,
maintainAspectRatio:false,
plugins:{
legend:{display:false},
tooltip:{
titleFont:{size:14, weight:"700"},
bodyFont:{size:13}
}
},
scales:{
y:{
min:0,
max:3,
grid:{color:"rgba(30,28,23,0.15)"},
ticks:{
stepSize:1,
font:{size:18, weight:"700"},
padding:8,
callback:(value)=>emotionText(["angry","sad","neutral","happy"][value] || "unknown")
}
},
x:{
grid:{display:false},
ticks:{
font:{size:15, weight:"600"},
padding:6
}
}
}
}

})

}else{

chart.data.datasets[0].data=values
chart.data.labels=values.map((_,i)=>i+1)

chart.update()

}

}


function uploadVideo(){

const file = uploadInput.files[0]

if(!file){
setStatus(t("status.chooseFileFirst"),"error")
return
}

const form = new FormData()
form.append("video",file)

setProgress(0)
uploadState.textContent = t("state.uploading")
setStatus(t("status.uploading"),"")

const xhr = new XMLHttpRequest()
xhr.open("POST",`${API_BASE}/analyze_video`)

xhr.upload.onprogress = (evt)=>{
if(evt.lengthComputable){
const percent = Math.round((evt.loaded / evt.total) * 100)
setProgress(percent)
}
}

xhr.onreadystatechange = ()=>{
if(xhr.readyState !== 4){
if(xhr.readyState === 2 || xhr.readyState === 3){
uploadState.textContent = t("state.processing")
setStatus(t("status.processing"),"")
}
return
}

if(xhr.status >= 200 && xhr.status < 300){
let data = {timeline:[]}

try{
data = JSON.parse(xhr.responseText)
}catch(_e){
setStatus(t("status.parseFail"),"error")
uploadState.textContent = t("state.error")
return
}

uploadTimeline = data.timeline || []
uploadAudioEmotion = data.audioEmotion || null
uploadFusionEmotion = data.fusedEmotion || null
uploadFusionConfidence = Number(data.fusedConfidence || 0)
drawDistribution(uploadTimeline)
drawUploadTrend(uploadTimeline)
uploadState.textContent = t("state.done")
resultState.textContent = t("state.readyResult")
setProgress(100)
if(data.audioError){
setStatus(t("status.videoDoneWithAudioFail", {count: data.timeline?.length || 0, error: data.audioError}),"error")
}else{
setStatus(t("status.videoDone", {count: data.timeline?.length || 0}),"ok")
}
}else{
uploadState.textContent = t("state.error")
resultState.textContent = t("state.error")
setStatus(t("status.uploadFail", {code: xhr.status}),"error")
}
}

xhr.onerror = ()=>{
uploadState.textContent = t("state.error")
resultState.textContent = t("state.error")
setStatus(t("status.networkFail"),"error")
}

xhr.send(form)

}


function analyzeTextInput(){
const text = (textInput?.value || "").trim()
if(!text){
textStatus.textContent = t("status.inputTextFirst")
textStatus.classList.remove("ok")
textStatus.classList.add("error")
return
}

textStatus.textContent = t("status.textAnalyzing")
textStatus.classList.remove("ok", "error")

fetch(`${API_BASE}/analyze_text`, {
method: "POST",
headers: {
"Content-Type": "application/json"
},
body: JSON.stringify({text})
})
.then(async r=>{
if(!r.ok){
let detail = ""
try{
const payload = await r.json()
detail = payload && payload.error ? `: ${payload.error}` : ""
}catch(_e){
detail = ""
}
throw new Error(`HTTP ${r.status}${detail}`)
}
return r.json()
})
.then(data=>{
uploadTextEmotion = data.textEmotion || "neutral"
statText.textContent = emotionText(uploadTextEmotion)
updateUploadCompanion(uploadFusionEmotion || uploadTextEmotion)
textStatus.textContent = t("status.textResult", {emotion: emotionText(uploadTextEmotion), confidence: data.confidence ?? "--"})
textStatus.classList.remove("error")
textStatus.classList.add("ok")
})
.catch(err=>{
textStatus.textContent = t("status.textFail", {message: err.message})
textStatus.classList.remove("ok")
textStatus.classList.add("error")
})
}


function drawDistribution(timelineData){

const count = {
angry:0,
sad:0,
neutral:0,
happy:0
}

timelineData.forEach(item=>{
if(item && count[item.emotion] !== undefined){
count[item.emotion] += 1
}
})

const labels = EMOTIONS
const values = labels.map(label=>count[label])
const total = values.reduce((sum, value)=>sum + value, 0)

const dominant = values.length ? labels[values.indexOf(Math.max(...values))] : "-"
const dominantRatio = total > 0 ? Math.round((Math.max(...values) / total) * 100) : 0

statTotal.textContent = String(total)
statDominant.textContent = dominant === "-" ? "-" : emotionText(dominant)
statRatio.textContent = `${dominantRatio}%`
statAudio.textContent = uploadAudioEmotion ? emotionText(uploadAudioEmotion) : "-"
statFusion.textContent = uploadFusionEmotion
? `${emotionText(uploadFusionEmotion)} (${Math.round(uploadFusionConfidence * 100)}%)`
: "-"
statText.textContent = uploadTextEmotion ? emotionText(uploadTextEmotion) : "-"
resultState.textContent = total > 0 ? t("state.analyzed") : t("state.noData")
updateUploadCompanion(uploadFusionEmotion || dominant)

if(distributionChart){
distributionChart.destroy()
}

distributionChart = new Chart(
document.getElementById("distributionChart"),
{
type:"bar",
data:{
labels,
datasets:[{
label:t("chart.count"),
data:values,
backgroundColor:["#d1495b", "#f79256", "#1f6f78", "#66a182"],
borderRadius:10,
maxBarThickness:58,
barPercentage:0.75,
categoryPercentage:0.7
}]
},
options:{
maintainAspectRatio:false,
layout:{padding:{left:10,right:10,top:6,bottom:8}},
plugins:{
legend:{display:false},
tooltip:{
titleFont:{size:15, weight:"700"},
bodyFont:{size:14}
}
},
scales:{
x:{
grid:{display:false},
ticks:{font:{size:18, weight:"700"}, padding:10}
},
y:{
beginAtZero:true,
ticks:{precision:0, font:{size:14, weight:"600"}, padding:8},
title:{display:true, text:t("chart.count"), font:{size:14, weight:"700"}}
}
}
}
}
)

}


function handleFilePreview(){

const file = uploadInput.files[0]

if(!file){
uploadPreview.style.display = "none"
uploadPreview.removeAttribute("src")
uploadTimeline = []
resetUploadInsights()
setStatus(t("status.waitingFile"),"")
uploadState.textContent = t("state.ready")
resultState.textContent = t("state.noData")
if(selectedFileText){
selectedFileText.textContent = t("upload.noFile")
}
setProgress(0)
return
}

const objectUrl = URL.createObjectURL(file)
uploadPreview.src = objectUrl
uploadPreview.style.display = "block"
uploadTimeline = []
resetUploadInsights()
uploadState.textContent = t("state.ready")
setStatus(t("status.selectedFile", {name: file.name}),"ok")
if(selectedFileText){
selectedFileText.textContent = file.name
}
setProgress(0)

}


function setProgress(percent){
const p = Math.max(0, Math.min(100, percent))
uploadProgressBar.style.width = `${p}%`
uploadProgressText.textContent = `${p}%`
}


function setStatus(message, type){
statusText.textContent = message
statusText.classList.remove("ok","error")
if(type){
statusText.classList.add(type)
}
}


function setLiveStatus(message, type){
liveStatusText.textContent = message
liveStatusText.classList.remove("ok","error")
if(type){
liveStatusText.classList.add(type)
}
}


function drawUploadTrend(timelineData){

const emotionMap = {
angry:0,
sad:1,
neutral:2,
happy:3
}

const points = timelineData
.filter(item=>item && item.emotion in emotionMap)
.map(item=>({
x:item.time,
y:emotionMap[item.emotion]
}))

if(uploadTrendChart){
uploadTrendChart.destroy()
}

uploadTrendChart = new Chart(
document.getElementById("uploadTrendChart"),
{
type:"line",
data:{
datasets:[{
label:t("chart.emotionLabel"),
data:points,
borderColor:"#1f6f78",
backgroundColor:"rgba(31,111,120,0.14)",
tension:0.28,
pointRadius:2,
fill:true
}]
},
options:{
responsive:true,
maintainAspectRatio:false,
plugins:{
legend:{display:false},
tooltip:{
titleFont:{size:15, weight:"700"},
bodyFont:{size:14}
}
},
scales:{
x:{
type:"linear",
title:{display:true,text:t("chart.time"), font:{size:15, weight:"700"}},
ticks:{font:{size:14, weight:"600"}, padding:8},
grid:{color:"rgba(30,28,23,0.12)"}
},
y:{
min:0,
max:3,
ticks:{
stepSize:1,
font:{size:18, weight:"700"},
padding:8,
callback:(value)=>emotionText(EMOTIONS[value]) || ""
},
grid:{color:"rgba(30,28,23,0.12)"}
}
}
}
}
)

}


function getSmoothedEmotion(currentEmotion){
if(!EMOTIONS.includes(currentEmotion)){
return "neutral"
}

liveVoteBuffer.push(currentEmotion)
if(liveVoteBuffer.length > LIVE_SMOOTH_WINDOW){
liveVoteBuffer.shift()
}

const counter = {
angry:0,
sad:0,
neutral:0,
happy:0
}

liveVoteBuffer.forEach(label=>{
counter[label] += 1
})

let bestEmotion = currentEmotion
let bestCount = -1

EMOTIONS.forEach(label=>{
if(counter[label] > bestCount){
bestCount = counter[label]
bestEmotion = label
}
})

return bestEmotion
}


function exportUploadJson(){
if(!uploadTimeline.length){
setStatus(t("status.noUploadToExport"), "error")
return
}

const payload = {
generatedAt: new Date().toISOString(),
totalPoints: uploadTimeline.length,
audioEmotion: uploadAudioEmotion,
fusedEmotion: uploadFusionEmotion,
textEmotion: uploadTextEmotion,
timeline: uploadTimeline
}

downloadBlob(
JSON.stringify(payload, null, 2),
`upload_emotion_report_${Date.now()}.json`,
"application/json"
)

setStatus(t("status.exportJsonOk"), "ok")
}


function exportUploadCsv(){
if(!uploadTimeline.length){
setStatus(t("status.noUploadToExport"), "error")
return
}

const rows = ["time,emotion"]
uploadTimeline.forEach(item=>{
rows.push(`${item.time},${item.emotion}`)
})

downloadBlob(
rows.join("\n"),
`upload_emotion_report_${Date.now()}.csv`,
"text/csv;charset=utf-8"
)

setStatus(t("status.exportCsvOk"), "ok")
}


function downloadBlob(content, filename, mimeType){
const blob = new Blob([content], {type:mimeType})
const link = document.createElement("a")
link.href = URL.createObjectURL(blob)
link.download = filename
document.body.appendChild(link)
link.click()
document.body.removeChild(link)
URL.revokeObjectURL(link.href)
}


function resetUploadInsights(){
statTotal.textContent = "0"
statDominant.textContent = "-"
statRatio.textContent = "0%"
statAudio.textContent = "-"
statFusion.textContent = "-"
statText.textContent = "-"
uploadAudioEmotion = null
uploadFusionEmotion = null
uploadFusionConfidence = 0
uploadTextEmotion = null
resetUploadCompanion()

if(distributionChart){
distributionChart.destroy()
distributionChart = null
}

if(uploadTrendChart){
uploadTrendChart.destroy()
uploadTrendChart = null
}
}


function drawFaceOverlay(faceBoxData, emotion){
if(!faceBoxData || !video.videoWidth || !video.videoHeight){
clearFaceOverlay()
return
}

const metrics = getVideoRenderMetrics()
if(!metrics){
clearFaceOverlay()
return
}

lastFaceBox = faceBoxData
lastFaceEmotion = emotion

const scaleX = metrics.width / video.videoWidth
const scaleY = metrics.height / video.videoHeight

faceOverlay.classList.remove("hidden")
faceBox.style.display = "block"
faceBox.style.left = `${metrics.offsetX + (faceBoxData.x * scaleX)}px`
faceBox.style.top = `${metrics.offsetY + (faceBoxData.y * scaleY)}px`
faceBox.style.width = `${faceBoxData.w * scaleX}px`
faceBox.style.height = `${faceBoxData.h * scaleY}px`
faceBoxLabel.textContent = emotion ? t("face.primaryWithEmotion", {emotion: emotionText(emotion)}) : t("face.primary")
}


function clearFaceOverlay(){
lastFaceBox = null
lastFaceEmotion = null
faceOverlay.classList.add("hidden")
faceBox.style.display = "none"
faceBox.style.width = "0"
faceBox.style.height = "0"
}


function redrawFaceOverlay(){
if(lastFaceBox){
drawFaceOverlay(lastFaceBox, lastFaceEmotion)
}
}


function getVideoRenderMetrics(){
const containerWidth = video.clientWidth
const containerHeight = video.clientHeight
const sourceWidth = video.videoWidth
const sourceHeight = video.videoHeight

if(!containerWidth || !containerHeight || !sourceWidth || !sourceHeight){
return null
}

const objectFit = window.getComputedStyle(video).objectFit || "fill"
let scale = 1

if(objectFit === "contain"){
scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight)
}else if(objectFit === "cover"){
scale = Math.max(containerWidth / sourceWidth, containerHeight / sourceHeight)
}else{
return {
width: containerWidth,
height: containerHeight,
offsetX: 0,
offsetY: 0
}
}

const renderedWidth = sourceWidth * scale
const renderedHeight = sourceHeight * scale

return {
width: renderedWidth,
height: renderedHeight,
offsetX: (containerWidth - renderedWidth) / 2,
offsetY: (containerHeight - renderedHeight) / 2
}
}


function getCompanionFeedback(emotion){
const normalized = EMOTIONS.includes(emotion) ? emotion : "neutral"
const presets = {
happy: {
moodClass: "mood-happy",
emotionText: t("companion.emotion.happy"),
message: t("companion.msg.happy"),
dialogs: t("companion.dialog.happy")
},
neutral: {
moodClass: "mood-neutral",
emotionText: t("companion.emotion.neutral"),
message: t("companion.msg.neutral"),
dialogs: t("companion.dialog.neutral")
},
sad: {
moodClass: "mood-sad",
emotionText: t("companion.emotion.sad"),
message: t("companion.msg.sad"),
dialogs: t("companion.dialog.sad")
},
angry: {
moodClass: "mood-angry",
emotionText: t("companion.emotion.angry"),
message: t("companion.msg.angry"),
dialogs: t("companion.dialog.angry")
}
}

return presets[normalized]
}


function pickRandomMessage(pool){
if(!Array.isArray(pool) || pool.length === 0){
return t("companion.defaultDialog")
}
return pool[Math.floor(Math.random() * pool.length)]
}


function getEmotionTrend(history){
if(!Array.isArray(history) || history.length < 4){
return "steady"
}

const tail = history.slice(-6)
const recent = tail.slice(-3)
const previous = tail.slice(0, Math.max(0, tail.length - 3))

const scoreMap = {
angry: -2,
sad: -1,
neutral: 0,
happy: 1
}

const avg = (arr)=> arr.reduce((sum, e)=>sum + (scoreMap[e] ?? 0), 0) / (arr.length || 1)
const delta = avg(recent) - avg(previous)

if(delta >= 0.55){
return "up"
}
if(delta <= -0.55){
return "down"
}
return "steady"
}


function getTrendDialog(trend, emotion){
if(trend === "up"){
return emotion === "happy"
? t("companion.trend.upHappy")
: t("companion.trend.up")
}
if(trend === "down"){
return t("companion.trend.down")
}
return null
}


function applyCompanionState(avatarEl, emotionEl, messageEl, dialogEl, historyBuffer, emotion){
if(!avatarEl || !emotionEl || !messageEl){
return
}

const payload = getCompanionFeedback(emotion)

if(Array.isArray(historyBuffer)){
historyBuffer.push(EMOTIONS.includes(emotion) ? emotion : "neutral")
if(historyBuffer.length > 12){
historyBuffer.shift()
}
}

const trend = getEmotionTrend(historyBuffer || [])
const trendDialog = getTrendDialog(trend, emotion)
const baseDialog = pickRandomMessage(payload.dialogs)

Object.values(COMPANION_MOOD_CLASS).forEach(className=>{
avatarEl.classList.remove(className)
})
avatarEl.classList.add(payload.moodClass)
emotionEl.textContent = payload.emotionText
messageEl.textContent = payload.message

if(dialogEl){
dialogEl.textContent = trendDialog || baseDialog
}
}


function updateRealtimeCompanion(emotion){
updateCompanion(emotion)
}


function updateUploadCompanion(emotion){
updateCompanion(emotion)
}


function updateCompanion(emotion){
applyCompanionState(
companionAvatar,
companionEmotion,
companionMessage,
companionDialog,
companionHistory,
emotion
)
}


function resetRealtimeCompanion(){
resetCompanion(t("companion.neutralPrompt"), t("companion.idleDialog"))
}


function resetUploadCompanion(){
resetCompanion(t("companion.neutralPrompt"), t("companion.idleDialog2"))
}


function resetCompanion(message, dialog){
if(companionAvatar && companionEmotion && companionMessage){
companionAvatar.classList.remove("mood-happy", "mood-sad", "mood-angry")
companionAvatar.classList.add("mood-neutral")
companionEmotion.textContent = currentLang === "en" ? "Current mood: --" : "当前情绪：--"
companionMessage.textContent = message
if(companionDialog){
companionDialog.textContent = dialog
}
companionHistory.length = 0
}
}


function initCompanionDrag(){
if(!floatingCompanion){
return
}

const rect = floatingCompanion.getBoundingClientRect()
setCompanionPosition(rect.left, rect.top, false)

const saved = loadCompanionPosition()
if(saved){
setCompanionPosition(saved.left, saved.top, false)
}

const dragTarget = companionDragHandle || floatingCompanion
dragTarget.addEventListener("pointerdown", onCompanionPointerDown)
window.addEventListener("resize", keepCompanionInViewport)
}


function onCompanionPointerDown(event){
if(!floatingCompanion){
return
}

if(event.pointerType !== "touch" && event.button !== 0){
return
}

const rect = floatingCompanion.getBoundingClientRect()
companionDragOffsetX = event.clientX - rect.left
companionDragOffsetY = event.clientY - rect.top
isCompanionDragging = true
floatingCompanion.classList.add("dragging")

if(companionDragHandle && companionDragHandle.setPointerCapture){
companionDragHandle.setPointerCapture(event.pointerId)
}

window.addEventListener("pointermove", onCompanionPointerMove)
window.addEventListener("pointerup", onCompanionPointerUp, {once:true})
event.preventDefault()
}


function onCompanionPointerMove(event){
if(!isCompanionDragging){
return
}

const left = event.clientX - companionDragOffsetX
const top = event.clientY - companionDragOffsetY
setCompanionPosition(left, top, true)
}


function onCompanionPointerUp(){
if(!floatingCompanion){
return
}

isCompanionDragging = false
floatingCompanion.classList.remove("dragging")
window.removeEventListener("pointermove", onCompanionPointerMove)

const rect = floatingCompanion.getBoundingClientRect()
saveCompanionPosition(rect.left, rect.top)
}


function setCompanionPosition(left, top, useTransition){
if(!floatingCompanion){
return
}

const width = floatingCompanion.offsetWidth || 320
const height = floatingCompanion.offsetHeight || 180
const maxLeft = Math.max(8, window.innerWidth - width - 8)
const maxTop = Math.max(8, window.innerHeight - height - 8)

const clampedLeft = Math.min(Math.max(8, left), maxLeft)
const clampedTop = Math.min(Math.max(8, top), maxTop)

floatingCompanion.style.right = "auto"
floatingCompanion.style.left = `${Math.round(clampedLeft)}px`
floatingCompanion.style.top = `${Math.round(clampedTop)}px`
floatingCompanion.style.transition = useTransition ? "none" : "left 160ms ease, top 160ms ease"
}


function keepCompanionInViewport(){
if(!floatingCompanion){
return
}

const rect = floatingCompanion.getBoundingClientRect()
setCompanionPosition(rect.left, rect.top, false)
}


function saveCompanionPosition(left, top){
try{
window.localStorage.setItem(COMPANION_POSITION_KEY, JSON.stringify({left, top}))
}catch(_e){
}
}


function loadCompanionPosition(){
try{
const raw = window.localStorage.getItem(COMPANION_POSITION_KEY)
if(!raw){
return null
}
const data = JSON.parse(raw)
if(typeof data.left === "number" && typeof data.top === "number"){
return data
}
return null
}catch(_e){
return null
}
}