"use client";

import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import { Copy, Save, Loader2, Search, FolderOpen } from 'lucide-react';
import { ShareMenu } from '@/components/Distribution/ShareMenu';
import Analytics from '@/lib/analytics';

interface SummaryUpdaterButtonGroupProps {
  isSaving: boolean;
  isDirty: boolean;
  onSave: () => Promise<void>;
  onCopy: () => Promise<void>;
  onFind?: () => void;
  onOpenFolder: () => Promise<void>;
  hasSummary: boolean;
  meetingId?: string;
  meetingTitle?: string;
  createdAt?: string;
  getMarkdown?: () => Promise<string>;
}

export function SummaryUpdaterButtonGroup({
  isSaving,
  isDirty,
  onSave,
  onCopy,
  onFind,
  onOpenFolder,
  hasSummary,
  meetingId,
  meetingTitle,
  createdAt,
  getMarkdown,
}: SummaryUpdaterButtonGroupProps) {
  return (
    <div className="flex items-center gap-1">
      <ButtonGroup>
        {/* Save button */}
        <Button
          variant="outline"
          size="sm"
          className={`${isDirty ? 'bg-green-200' : ""}`}
          title={isSaving ? "Saving" : "Save Changes"}
          onClick={() => {
            Analytics.trackButtonClick('save_changes', 'meeting_details');
            onSave();
          }}
          disabled={isSaving}
        >
          {isSaving ? (
            <>
              <Loader2 className="animate-spin" />
              <span className="hidden lg:inline">Saving...</span>
            </>
          ) : (
            <>
              <Save />
              <span className="hidden lg:inline">Save</span>
            </>
          )}
        </Button>

        {/* Copy button */}
        <Button
          variant="outline"
          size="sm"
          title="Copy Summary"
          onClick={() => {
            Analytics.trackButtonClick('copy_summary', 'meeting_details');
            onCopy();
          }}
          disabled={!hasSummary}
          className="cursor-pointer"
        >
          <Copy />
          <span className="hidden lg:inline">Copy</span>
        </Button>
      </ButtonGroup>

      {/* Share menu */}
      {hasSummary && meetingId && meetingTitle && getMarkdown && (
        <ShareMenu
          meetingId={meetingId}
          meetingTitle={meetingTitle}
          createdAt={createdAt}
          getMarkdown={getMarkdown}
        />
      )}
    </div>
  );
}
