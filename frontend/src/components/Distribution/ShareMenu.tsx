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
import { Share2, FileDown, Link2, Mail, Loader2, Send } from 'lucide-react';
import { toast } from 'sonner';
import { exportSummaryToPdf } from '@/services/pdfExportService';
import { getMeeting } from '@/services/firestoreService';
import { DistributionDialog } from './DistributionDialog';

interface ShareMenuProps {
  meetingId: string;
  meetingTitle: string;
  createdAt?: string;
  getMarkdown: () => Promise<string>;
}

export function ShareMenu({ meetingId, meetingTitle, createdAt, getMarkdown }: ShareMenuProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [showDistribution, setShowDistribution] = useState(false);
  const [cachedMarkdown, setCachedMarkdown] = useState('');

  const handleExportPdf = async () => {
    setIsExporting(true);
    try {
      const markdown = await getMarkdown();
      if (!markdown || !markdown.trim()) {
        toast.error('Aucun contenu à exporter');
        return;
      }

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
    const link = `https://app.remedee.com/cr/${meetingId}`;
    try {
      await navigator.clipboard.writeText(link);
      toast.success('Lien copié dans le presse-papiers');
    } catch {
      toast.error('Impossible de copier le lien');
    }
  };

  const handleOpenDistribution = async () => {
    try {
      const markdown = await getMarkdown();
      setCachedMarkdown(markdown);
      setShowDistribution(true);
    } catch (err) {
      console.error('Failed to get markdown for distribution:', err);
      toast.error('Erreur lors de la récupération du contenu');
    }
  };

  return (
    <>
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
          <DropdownMenuItem onClick={handleOpenDistribution}>
            <Send className="w-4 h-4 mr-2" />
            Distribuer...
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <DistributionDialog
        open={showDistribution}
        onOpenChange={setShowDistribution}
        meetingId={meetingId}
        meetingTitle={meetingTitle}
        createdAt={createdAt}
        markdown={cachedMarkdown}
      />
    </>
  );
}
