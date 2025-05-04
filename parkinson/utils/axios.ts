import axios from "axios";

// IOS: use localhost, NOT MAC OS: use ip address

const baseURL = "http://192.168.1.40:8000/api"

export const PUBLIC_URL = "http://192.168.1.40:8000"

export default axios.create({
  baseURL: baseURL,
});

