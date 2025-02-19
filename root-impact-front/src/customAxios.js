import axios from "axios";

export const customAxios = axios.create({
  baseURL: `${process.env.REACT_APP_API}`,
  headers: {
    "content-type": "application/json",
  },
});
