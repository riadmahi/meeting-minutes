import jsPDF from 'jspdf';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

interface ExportOptions {
  title: string;
  markdown: string;
  meetingId: string;
  createdAt?: string;
  participants?: string[];
}

/**
 * Exports a meeting summary as a PDF document.
 * Parses markdown into structured sections and renders them with jsPDF.
 */
export async function exportSummaryToPdf({
  title,
  markdown,
  meetingId,
  createdAt,
  participants,
}: ExportOptions): Promise<void> {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  const checkPageBreak = (needed: number) => {
    const pageHeight = doc.internal.pageSize.getHeight();
    if (y + needed > pageHeight - margin) {
      doc.addPage();
      y = margin;
    }
  };

  // Title
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  const titleLines = doc.splitTextToSize(title, contentWidth);
  checkPageBreak(titleLines.length * 8);
  doc.text(titleLines, margin, y);
  y += titleLines.length * 8 + 2;

  // Metadata
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(120, 120, 120);

  if (createdAt) {
    const dateStr = format(new Date(createdAt), "d MMMM yyyy 'à' HH:mm", { locale: fr });
    doc.text(`Date : ${dateStr}`, margin, y);
    y += 5;
  }

  if (participants && participants.length > 0) {
    doc.text(`Participants : ${participants.join(', ')}`, margin, y);
    y += 5;
  }

  doc.text(`ID : ${meetingId}`, margin, y);
  y += 8;

  // Separator
  doc.setDrawColor(200, 200, 200);
  doc.line(margin, y, pageWidth - margin, y);
  y += 6;

  // Reset text color
  doc.setTextColor(0, 0, 0);

  // Parse markdown into lines
  const lines = markdown.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      y += 3;
      continue;
    }

    // H2 heading
    if (trimmed.startsWith('## ')) {
      checkPageBreak(12);
      y += 4;
      doc.setFontSize(13);
      doc.setFont('helvetica', 'bold');
      const heading = trimmed.replace(/^## /, '');
      const headingLines = doc.splitTextToSize(heading, contentWidth);
      doc.text(headingLines, margin, y);
      y += headingLines.length * 6 + 3;
      continue;
    }

    // H3 heading
    if (trimmed.startsWith('### ')) {
      checkPageBreak(10);
      y += 2;
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      const heading = trimmed.replace(/^### /, '');
      const headingLines = doc.splitTextToSize(heading, contentWidth);
      doc.text(headingLines, margin, y);
      y += headingLines.length * 5 + 2;
      continue;
    }

    // Table row (skip header separators)
    if (trimmed.startsWith('|') && trimmed.includes('---')) {
      continue;
    }

    if (trimmed.startsWith('|')) {
      checkPageBreak(6);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      // Simplify table rows into text
      const cells = trimmed
        .split('|')
        .filter((c) => c.trim())
        .map((c) => c.trim());
      const rowText = cells.join('  |  ');
      const rowLines = doc.splitTextToSize(rowText, contentWidth);
      doc.text(rowLines, margin, y);
      y += rowLines.length * 4 + 1;
      continue;
    }

    // Bullet point
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      checkPageBreak(6);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      const bullet = trimmed.replace(/^[-*] /, '');
      const bulletLines = doc.splitTextToSize(bullet, contentWidth - 6);
      doc.text('•', margin, y);
      doc.text(bulletLines, margin + 6, y);
      y += bulletLines.length * 4.5 + 1;
      continue;
    }

    // Bold text (simplified — render as bold paragraph)
    if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
      checkPageBreak(6);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      const bold = trimmed.replace(/^\*\*/, '').replace(/\*\*$/, '');
      const boldLines = doc.splitTextToSize(bold, contentWidth);
      doc.text(boldLines, margin, y);
      y += boldLines.length * 4.5 + 1;
      continue;
    }

    // Regular paragraph
    checkPageBreak(6);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    // Strip inline markdown formatting for clean PDF
    const clean = trimmed.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1');
    const paraLines = doc.splitTextToSize(clean, contentWidth);
    doc.text(paraLines, margin, y);
    y += paraLines.length * 4.5 + 1;
  }

  // Footer on every page
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    const pageHeight = doc.internal.pageSize.getHeight();
    doc.text(`Généré par Remedee — Page ${i}/${pageCount}`, margin, pageHeight - 10);
  }

  // Save
  const safeTitle = title.replace(/[^a-zA-Z0-9àâäéèêëïîôùûüÿç _-]/g, '').substring(0, 50);
  doc.save(`${safeTitle}.pdf`);
}
