import * as fs from 'fs';

const path = "C:\\Users\\BIZRA-OS\\Downloads\\extracted-history\\conversations.json";
const raw = fs.readFileSync(path, 'utf-8');
const data = JSON.parse(raw);

if (Array.isArray(data) && data.length > 0) {
  const nonEmpty = data.find(d => d.chat_messages && d.chat_messages.length > 0);
  if (nonEmpty) {
    console.log("Found conversation with messages:", nonEmpty.uuid);
    console.log("Message Sample:", JSON.stringify(nonEmpty.chat_messages[0], null, 2).substring(0, 500));
  } else {
    console.log("No conversations with messages found.");
  }
} else {
  console.log("Data is not an array or is empty.");
}
