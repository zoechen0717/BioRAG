from core.document_processor import DocumentProcessor
from core.base_rag import BaseRAG
import logging

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 初始化处理器和RAG系统
    processor = DocumentProcessor()
    rag = BaseRAG("config.yaml")
    
    # 处理论文
    papers_dir = "data/papers"
    logger.info(f"开始处理 {papers_dir} 中的论文...")
    
    # 查找所有PDF文件
    pdf_files = processor.find_pdf_files(papers_dir)
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        try:
            logger.info(f"正在处理: {pdf_file}")
            content, metadata = processor.process_pdf(pdf_file)
            rag.add_document(content, metadata)
            logger.info(f"成功添加: {pdf_file}")
        except Exception as e:
            logger.error(f"处理失败 {pdf_file}: {str(e)}")
            continue
    
    # 保存RAG系统
    logger.info("保存RAG系统...")
    rag.save("bio_rag.pkl")
    logger.info("完成!")

if __name__ == "__main__":
    main() 