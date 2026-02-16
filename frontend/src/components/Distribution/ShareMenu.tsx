'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Share2, FileDown, Link2, Mail, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { exportSummaryToPdf } from '@/services/pdfExportService';
import { getMeeting } from '@/services/firestoreService';

interface ShareMenuProps {
  meetingId: string;
  meetingTitle: string;
  createdAt?: string;
  getMarkdown: () => Promise<string>;
}

export function ShareMenu({ meetingId, meetingTitle, createdAt, getMarkdown }: ShareMenuProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExportPdf = async () => {
    setIsExporting(true);
    try {
      const markdown = await getMarkdown();
      if (!markdown || !markdown.trim()) {
        toast.error('Aucun contenu à exporter');
        return;
      }

      // Fetch meeting to get participants
      const meeting = await getMeeting(meetingId).catch(() => null);

      await exportSummaryToPdf({
        title: meetingTitle,
        markdown,
        meetingId,
        createdAt,
        participants: meeting?.participants,
      });

      toast.success('PDF exporté avec succès');
    } catch (err) {
      console.error('PDF export failed:', err);
      toast.error("Erreur lors de l'export PDF");
    } finally {
      setIsExporting(false);
    }
  };

  const handleCopyLink = async () => {
    // Build a shareable link — for now the Firestore meeting ID
    // In production this would be a web app URL like https://app.remedee.com/cr/{id}
    const link = `https://app.remedee.com/cr/${meetingId}`;
    try {
      await navigator.clipboard.writeText(link);
      toast.success('Lien copié dans le presse-papiers');
    } catch {
      toast.error('Impossible de copier le lien');
    }
  };

  const handleEmailShare = async () => {
    try {
      const markdown = await getMarkdown();
      const subject = encodeURIComponent(`Compte-rendu : ${meetingTitle}`);
      const body = encodeURIComponent(
        `Bonjour,\n\nVoici le compte-rendu de la réunion "${meetingTitle}".\n\n${markdown.substring(0, 2000)}${markdown.length > 2000 ? '\n\n[Contenu tronqué — voir le CR complet dans l\'app]' : ''}\n\n---\nGénéré par Remedee`,
      );
      window.open(`mailto:?subject=${subject}&body=${body}`, '_blank');
    } catch (err) {
      console.error('Email share failed:', err);
      toast.error("Erreur lors du partage par email");
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={isExporting}>
          {isExporting ? (
            <Loader2 className="animate-spin" />
          ) : (
            <Share2 />
          )}
          <span className="hidden lg:inline">Partager</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem onClick={handleExportPdf}>
          <FileDown className="w-4 h-4 mr-2" />
          Exporter en PDF
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleCopyLink}>
          <Link2 className="w-4 h-4 mr-2" />
          Copier le lien
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleEmailShare}>
          <Mail className="w-4 h-4 mr-2" />
          Envoyer par email
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
