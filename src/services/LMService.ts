import OpenAI from 'openai';
import { ChatMessage, ChatCompletionResponse } from '../types/chat';
import { GenericDBService } from './GenericDBService';
import { config } from '../config';
import { Logger } from '../utils/logger';

export class LMService {
  private openai: OpenAI;
  private dbService: GenericDBService;

  constructor() {
    this.openai = new OpenAI({
      apiKey: config.openai.apiKey,
    });
    this.dbService = new GenericDBService();
  }

  /**
   * Complete chat with OpenAI using messages array and handle tool calling internally
   */
  async completion(
    messages: ChatMessage[]
  ): Promise<{ messages: ChatMessage[]; success: boolean; finalResult?: any }> {
    Logger.debug('🤖 Processing completion with model:', config.openai.model);

    //! Definir el array de tools que contenga la definición de la función save-email
    //! Se debe definir el nombre, argumentos, descripción, esquema de parámetros y retorno
    const tools: any = [];

    let maxRetries = 3;
    let currentRetry = 0;
    let hasToolCalls = true;
    const processedCallIds = new Set<string>();
    const conversationMessages = [...messages]; // Copy initial messages
    let finalResult: any = null;

    do {
      // Convert our ChatMessage type to OpenAI format
      const openaiMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] =
        conversationMessages.map((msg) => {
          if (msg.role === 'tool') {
            return {
              role: 'tool',
              content: msg.content,
              tool_call_id: msg.tool_call_id!,
            };
          }

          if (msg.tool_calls) {
            return {
              role: msg.role as 'assistant',
              content: msg.content,
              tool_calls: msg.tool_calls.map((tc) => ({
                id: tc.id,
                type: tc.type,
                function: tc.function,
              })),
            };
          }

          return {
            role: msg.role as 'system' | 'user' | 'assistant',
            content: msg.content,
          };
        });

      //! Usar el método create de chat.completions para llamar al modelo con los items de messages (tools calls o messages normales)
      //! El modelo debe ser usado de manera determinista
      const completion: ChatCompletionResponse = {} as any; // Reemplazar con la llamada real

      const assistantMessage = completion.choices[0].message;

      // Add assistant message to conversation
      conversationMessages.push({
        role: 'assistant',
        content: assistantMessage.content || '',
        tool_calls: assistantMessage.tool_calls?.map((tc) => ({
          id: tc.id,
          type: tc.type,
          function: tc.function,
        })),
      });

      // Check if there are tool calls
      if (
        !assistantMessage.tool_calls ||
        assistantMessage.tool_calls.length === 0
      ) {
        hasToolCalls = false;
        // Capture the final LLM response
        finalResult = {
          content: assistantMessage.content || '',
          role: 'assistant',
        };
        break;
      }

      // Process each tool call
      for (const toolCall of assistantMessage.tool_calls) {
        // Avoid duplicate calls
        if (processedCallIds.has(toolCall.id)) {
          Logger.debug(`⏭️ Skipping duplicate tool call: ${toolCall.id}`);
          continue;
        }
        processedCallIds.add(toolCall.id);

        // Call the function and get result
        let toolResult: any;
        try {
          toolResult = await this.callFunction(
            toolCall.function.name,
            toolCall.function.arguments
          );
        } catch (error) {
          currentRetry++;
        }

        // Add tool result to conversation
        conversationMessages.push({
          role: 'tool',
          content: JSON.stringify(toolResult),
          tool_call_id: toolCall.id,
        });
      }

      // Check if we should retry due to errors
      if (currentRetry >= maxRetries) {
        Logger.warn(`⚠️ Max retries reached`);
        hasToolCalls = false;
      }
    } while (hasToolCalls && currentRetry < maxRetries);

    return {
      messages: conversationMessages,
      success: finalResult !== null,
      finalResult,
    };
  }

  /**
   * Call a function by name and handle errors appropriately
   */
  async callFunction(
    functionName: string,
    argumentsString: string
  ): Promise<any> {
    try {
      //! Implementar el llamado de la tool según su nombre
      //! Parsear los argumentos desde argumentsString
    } catch (error) {
      return {
        success: false,
        error: `Tool call failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
    }
  }

  /**
   * Get current model being used
   */
  getCurrentModel(): string {
    return config.openai.model;
  }
}
