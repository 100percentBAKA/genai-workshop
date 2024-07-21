import { useMutation } from "@tanstack/react-query";
import axios from "axios";

const axiosInstance = axios.create({
  baseURL: "http://localhost:8000",
});

async function postQuery(body: { query: string }) {
  return axiosInstance.post("/query", body);
}

export function usePostQuery() {
  return useMutation({
    mutationFn: (body: { query: string }) => postQuery(body),
  });
}
