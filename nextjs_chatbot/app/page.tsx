"use client";

import { Button } from "../components/ui/button";
import { Input } from "@/components/ui/input";
import React, { useState, useEffect, useRef } from "react";
import { IoSend } from "react-icons/io5";
import ReactMarkdown from "react-markdown";
import { RiRobot2Line } from "react-icons/ri";
import { Skeleton } from "@/components/ui/skeleton";
import { usePostQuery } from "@/lib/mutations";
import { useToast } from "@/components/ui/use-toast";
import { ToastAction } from "@/components/ui/toast";

function Chat() {
  const { toast } = useToast();

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [loading, setLoading] = useState(false);
  const [toastDisplayed, setToastDisplayed] = useState(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const lastMessageRef = useRef<HTMLDivElement>(null);

  const { mutate: postQuery, isPending } = usePostQuery();

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInput(event.target.value);
  };

  const handleToast = () => {
    console.log("handling toast");
    !toastDisplayed &&
      toast({
        title: "Wait for 50 seconds",
        description:
          "The backend of this project is deployed on Render, the free tier of Render puts the application to sleep due to the inactivity, please wait for 50 or more seconds to get the response",
        action: <ToastAction altText="close">Close</ToastAction>,
      });

    //   after the toast is displayed, set the toastDisplayed state to true
    setToastDisplayed(true);
    console.log("toast handling done");
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    handleToast();

    if (input.trim() === "") return;

    setLoading(true);

    const userMessage = { role: "user", content: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    const reqBody = {
      query: input,
    };

    postQuery(reqBody, {
      onSuccess: (response) => {
        const botMessage = {
          role: "bot",
          content: response.data.response,
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
        setLoading(false);
      },
      onError: (error) => {
        const botMessage = {
          role: "bot",
          content: "An error occurred. Please try again.",
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
        setLoading(false);
      },
    });

    setInput("");
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
  };

  const suggestions = [
    "Hello world program in C#?",
    "CRUD operations using fastapi",
    "SQL queries for student database",
    "Frequently used git commands",
  ];

  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    const lastMessage = lastMessageRef.current;

    if (chatContainer && lastMessage) {
      const containerHeight = chatContainer.clientHeight;
      const messageHeight = lastMessage.offsetTop + lastMessage.clientHeight;

      if (messageHeight > containerHeight) {
        chatContainer.scrollTop = messageHeight - containerHeight;
      }
    }
  }, [messages]);

  return (
    <section className="max-w-7xl w-full h-full flex flex-col px-4 md:px-8 mx-auto">
      <div className="rounded-md border border-gray-200 w-full h-full mt-8 flex flex-col">
        {/* Suggestion Windows */}
        <div className="grid grid-cols-1 lg:grid-cols-2 items-center p-3 gap-3 border-b border-gray-200">
          {suggestions.map((text, index) => (
            <div
              key={index}
              className="border border-gray-200 p-2 rounded-md cursor-pointer transition-colors duration-150 hover:border-blue-300"
              onClick={() => handleSuggestionClick(text)}
            >
              {text}
            </div>
          ))}
        </div>

        {/* Chat Window */}
        <div
          className="flex-1 p-4 overflow-y-auto flex flex-col min-h-[550px] max-h-[600px]"
          ref={chatContainerRef}
        >
          {messages.map((message, index) => (
            <div
              key={index}
              className={`my-2 max-w-[250px] lg:max-w-[500px] xl:max-w-[550px] p-2 rounded-md ${
                message.role === "user"
                  ? "bg-blue-100 self-end"
                  : "bg-gray-100 self-start"
              }`}
              ref={index === messages.length - 1 ? lastMessageRef : null}
            >
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ))}
          {loading && (
            <Skeleton className="my-2 max-w-[250px] p-2 rounded-md min-h-[40px]" />
          )}

          {/* Fall back for empty chat window */}
          <div className="flex-1 w-full flex items-center justify-center">
            {messages.length === 0 && !loading && (
              <div className="flex flex-col space-y-4 items-center">
                <RiRobot2Line className="w-[24px] h-[24px]" />
                <p>Send a message to start chatting with my portfolio</p>
                <p>Feel free to use the above suggestions</p>
              </div>
            )}
          </div>
        </div>

        {/* Input Field goes here . . . */}
        <div className="p-4 border-t border-gray-200">
          <form className="flex items-center space-x-2" onSubmit={handleSubmit}>
            <Input
              type="text"
              value={input}
              onChange={handleInputChange}
              className="flex-1 border border-gray-300 p-2 rounded-md"
              placeholder="Type a message..."
            />
            <Button
              className="rounded-md flex items-center"
              type="submit"
              disabled={isPending}
            >
              <span className="hidden lg:flex">Send</span>
              <IoSend className="lg:hidden" />
            </Button>
          </form>
        </div>
      </div>
    </section>
  );
}

export default Chat;
