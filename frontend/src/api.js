import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000/api';

export const getStatus = () => axios.get(`${API_URL}/system/status`).then(res => res.data);
export const getCurrentMetrics = () => axios.get(`${API_URL}/metrics/current`).then(res => res.data);
export const getTimeline = () => axios.get(`${API_URL}/timeline`).then(res => res.data);
export const getLatestPrediction = () => axios.get(`${API_URL}/predictions/latest`).then(res => res.data);
export const runSimulationStep = () => axios.post(`${API_URL}/scale/run`).then(res => res.data);
export const trainModels = () => axios.post(`${API_URL}/model/train`).then(res => res.data);
