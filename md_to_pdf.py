#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to PDF converter script
将 Markdown 文件转换为 PDF
"""

import markdown
import weasyprint
import os
import sys
from pathlib import Path

def markdown_to_pdf(md_file_path, output_pdf_path=None, css_style=None):
    """
    将 Markdown 文件转换为 PDF
    
    Args:
        md_file_path (str): Markdown 文件路径
        output_pdf_path (str): 输出 PDF 文件路径，如果为 None 则自动生成
        css_style (str): 自定义 CSS 样式
    
    Returns:
        str: 生成的 PDF 文件路径
    """
    
    # 读取 Markdown 文件
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 转换 Markdown 为 HTML
    md = markdown.Markdown(extensions=[
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'markdown.extensions.toc',
        'markdown.extensions.meta'
    ])
    html_content = md.convert(md_content)
    
    # 默认 CSS 样式
    default_css = """
    @page {
        size: A4;
        margin: 2cm;
    }
    
    body {
        font-family: "SimSun", "Microsoft YaHei", Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    
    h1 { font-size: 24pt; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
    h2 { font-size: 20pt; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; }
    h3 { font-size: 16pt; }
    h4 { font-size: 14pt; }
    h5 { font-size: 12pt; }
    h6 { font-size: 11pt; }
    
    p {
        margin-bottom: 1em;
        text-align: justify;
    }
    
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: "Consolas", "Monaco", "Courier New", monospace;
        font-size: 10pt;
    }
    
    pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1em;
        overflow-x: auto;
        margin: 1em 0;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    
    table, th, td {
        border: 1px solid #ddd;
    }
    
    th, td {
        padding: 8px;
        text-align: left;
    }
    
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    
    ul, ol {
        margin: 1em 0;
        padding-left: 2em;
    }
    
    li {
        margin-bottom: 0.5em;
    }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 1em 0;
        padding-left: 1em;
        color: #666;
        font-style: italic;
    }
    
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
    }
    
    a {
        color: #3498db;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    """
    
    # 使用自定义 CSS 或默认 CSS
    css_content = css_style if css_style else default_css
    
    # 构建完整的 HTML 文档
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <style>
        {css_content}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # 生成输出文件路径
    if output_pdf_path is None:
        md_path = Path(md_file_path)
        output_pdf_path = md_path.with_suffix('.pdf')
    
    # 转换为 PDF
    try:
        weasyprint.HTML(string=full_html).write_pdf(output_pdf_path)
        print(f"✅ 成功转换: {md_file_path} -> {output_pdf_path}")
        return str(output_pdf_path)
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return None

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python md_to_pdf.py <markdown_file> [output_pdf_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(md_file):
        print(f"❌ 文件不存在: {md_file}")
        sys.exit(1)
    
    result = markdown_to_pdf(md_file, output_file)
    if result:
        print(f"📄 PDF 文件已生成: {result}")
    else:
        print("❌ PDF 生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main()