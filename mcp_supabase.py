#!/usr/bin/env python3
"""
MCP Server for 財報 RAG 向量搜尋
使用 stdio 模式運行，提供文字搜尋和向量搜尋功能
"""

import os
import json
import sys
import logging
from typing import Any, Sequence
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 設定日誌記錄（方便除錯）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 載入 .env 檔案中的環境變數
load_dotenv()

# 從環境變數讀取 Supabase 設定
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
TABLE_NAME = os.getenv('TABLE_NAME', 'rag_chunks')  # 預設表名為 rag_chunks

# 從環境變數讀取 OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 驗證必要的環境變數是否存在
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("錯誤：請在 .env 檔案中設定 SUPABASE_URL 和 SUPABASE_KEY")
    sys.exit(1)

if not OPENAI_API_KEY:
    logger.error("錯誤：請在 .env 檔案中設定 OPENAI_API_KEY")
    sys.exit(1)

# 初始化 Supabase 客戶端
# 這會建立一個連接到你的 Supabase 專案的客戶端
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase 客戶端初始化成功")
except Exception as e:
    logger.error(f"初始化 Supabase 客戶端失敗: {e}")
    sys.exit(1)

# 初始化 OpenAI 客戶端（用於產生 embedding）
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI 客戶端初始化成功")
except Exception as e:
    logger.error(f"初始化 OpenAI 客戶端失敗: {e}")
    sys.exit(1)

# 建立 MCP Server 實例
# 這是 MCP 伺服器的主要物件，用來註冊工具和處理請求
app = Server("supabase-rag-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    列出所有可用的工具
    這個函數會告訴 Claude/ChatGPT 有哪些工具可以使用
    """
    return [
        Tool(
            name="search_text",
            description="財報文字關鍵字搜尋：使用 SQL LIKE 查詢在財報內容中搜尋關鍵字",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "要搜尋的關鍵字"
                    }
                },
                "required": ["keyword"]
            }
        ),
        Tool(
            name="vector_search",
            description="財報向量相似度搜尋：使用 OpenAI embedding 和 pgvector 進行語義搜尋",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "使用者輸入的自然語言問題"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相似的前幾筆結果",
                        "default": 5
                    }
                },
                "required": ["query_text"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """
    處理工具呼叫
    當 Claude/ChatGPT 要使用某個工具時，會呼叫這個函數
    """
    try:
        if name == "search_text":
            # 工具 1：文字關鍵字搜尋
            keyword = arguments.get("keyword")
            if not keyword:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "缺少必要參數：keyword"}, ensure_ascii=False)
                )]
            
            logger.info(f"執行文字搜尋，關鍵字：{keyword}")
            
            # 使用 Supabase 的 ilike 方法進行模糊搜尋
            # ilike 是 PostgreSQL 的不區分大小寫 LIKE 查詢
            # f"%{keyword}%" 表示在 content 欄位中搜尋包含 keyword 的內容
            response = supabase.table(TABLE_NAME)\
                .select("id, doc_id, chunk_id, content")\
                .ilike("content", f"%{keyword}%")\
                .execute()
            
            # 回傳搜尋結果
            result = {
                "success": True,
                "count": len(response.data),
                "results": response.data
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        
        elif name == "vector_search":
            # 工具 2：向量相似度搜尋
            query_text = arguments.get("query_text")
            top_k = arguments.get("top_k", 5)  # 預設返回 5 筆結果
            
            if not query_text:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "缺少必要參數：query_text"}, ensure_ascii=False)
                )]
            
            logger.info(f"執行向量搜尋，查詢：{query_text}，top_k：{top_k}")
            
            # 步驟 1：使用 OpenAI 將查詢文字轉換成向量
            # text-embedding-3-small 模型會產生 1536 維的向量
            # 注意：Supabase 的 embedding 欄位維度必須與此相同（1536），否則查詢會失敗
            try:
                embedding_response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query_text
                )
                query_embedding = embedding_response.data[0].embedding
                logger.info(f"成功產生 embedding，維度：{len(query_embedding)}")
            except Exception as e:
                logger.error(f"產生 embedding 失敗: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"產生 embedding 失敗: {str(e)}"}, ensure_ascii=False)
                )]
            
            # 步驟 2：在 Supabase 中進行向量相似度搜尋
            # 使用 pgvector 的 <-> 運算子（L2 距離）或 <=> 運算子（cosine 距離）
            # 這裡使用 <-> 運算子，距離越小表示越相似
            try:
                # 使用 Supabase 的 RPC 函數或直接 SQL 查詢
                # 方法 1：如果 Supabase 有建立 match_documents 函數，可以使用：
                # response = supabase.rpc('match_documents', {
                #     'query_embedding': query_embedding,
                #     'match_count': top_k
                # }).execute()
                
                # 方法 2：直接使用 SQL 查詢（推薦，因為不需要額外建立函數）
                # 使用 PostgREST 的 select 配合 order 和 limit
                # 注意：Supabase Python SDK 可能不直接支援向量運算子，所以我們用 RPC 或原始 SQL
                
                # 這裡我們使用一個通用的方法：透過 RPC 函數
                # 如果沒有 RPC 函數，我們需要先建立一個
                # 為了簡化，我們假設使用 match_documents RPC 函數
                # 如果沒有，請在 Supabase SQL Editor 中執行以下 SQL 建立函數：
                """
                CREATE OR REPLACE FUNCTION match_documents(
                    query_embedding vector(1536),
                    match_count int DEFAULT 5
                )
                RETURNS TABLE (
                    id bigint,
                    doc_id text,
                    chunk_id text,
                    content text,
                    similarity float
                )
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN QUERY
                    SELECT
                        rag_chunks.id,
                        rag_chunks.doc_id,
                        rag_chunks.chunk_id,
                        rag_chunks.content,
                        1 - (rag_chunks.embedding <=> query_embedding) as similarity
                    FROM rag_chunks
                    ORDER BY rag_chunks.embedding <=> query_embedding
                    LIMIT match_count;
                END;
                $$;
                """
                
                # 嘗試使用 RPC 函數
                try:
                    response = supabase.rpc('match_documents', {
                        'query_embedding': query_embedding,
                        'match_count': top_k
                    }).execute()
                    
                    results = response.data if hasattr(response, 'data') else []
                    
                except Exception as rpc_error:
                    # 如果 RPC 函數不存在，使用替代方法：直接 SQL 查詢
                    logger.warning(f"RPC 函數不存在，嘗試使用替代方法: {rpc_error}")
                    
                    # 使用 Supabase 的 postgrest 查詢
                    # 注意：這需要 embedding 欄位有建立索引才能高效查詢
                    # 這裡我們用一個簡化的方法：先取得所有資料，然後在 Python 中計算相似度
                    # （這不是最佳做法，但可以運作）
                    
                    # 更好的做法是在 Supabase 建立 match_documents 函數
                    logger.error("請在 Supabase 中建立 match_documents RPC 函數，參考程式碼註解")
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "向量搜尋需要先在 Supabase 建立 match_documents RPC 函數",
                            "hint": "請參考程式碼中的 SQL 註解建立函數"
                        }, ensure_ascii=False)
                    )]
                
                # 格式化結果
                result = {
                    "success": True,
                    "query": query_text,
                    "count": len(results),
                    "results": results
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"向量搜尋失敗: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"向量搜尋失敗: {str(e)}"}, ensure_ascii=False)
                )]
        
        else:
            # 如果工具名稱不存在
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"未知的工具：{name}"}, ensure_ascii=False)
            )]
    
    except Exception as e:
        # 捕捉所有未預期的錯誤，避免程式崩潰
        logger.error(f"處理工具呼叫時發生錯誤: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"處理請求時發生錯誤: {str(e)}"}, ensure_ascii=False)
        )]


async def main():
    """
    主函數：啟動 MCP Server
    使用 stdio 模式，透過標準輸入輸出與 Claude/ChatGPT 通訊
    """
    # 使用 stdio_server 啟動伺服器
    # 這會讓 MCP Server 透過標準輸入輸出（stdin/stdout）與外部程式通訊
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    # 啟動 MCP Server
    # 使用 asyncio 運行異步主函數
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到中斷訊號，正在關閉伺服器...")
    except Exception as e:
        logger.error(f"伺服器執行錯誤: {e}", exc_info=True)
        sys.exit(1)


