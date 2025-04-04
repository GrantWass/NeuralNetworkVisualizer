import React from "react";
import { Button } from "@/components/ui/button";

interface InfoPopupProps {
  title: string;
  message: string;
  onClose: () => void;
}

const InfoPopup: React.FC<InfoPopupProps> = ({ title, message, onClose }) => {
  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded-lg max-w-lg w-full">
        <h3 className="text-xl font-semibold mb-4">{title}</h3>
        <p className="whitespace-pre-wrap mb-4">{message}</p>
        <div className="flex justify-end">
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
};

export default InfoPopup;
