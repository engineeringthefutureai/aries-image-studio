import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 m-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-800 dark:text-rose-300">
          <div className="flex items-center gap-2 mb-2 font-semibold">
            <AlertCircle className="w-5 h-5 text-rose-600" />
            <span>{this.props.fallbackTitle || 'Component rendering error'}</span>
          </div>
          <p className="text-xs font-mono mb-4 text-rose-700 dark:text-rose-400">
            {this.state.error?.message || 'Unknown render error'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-600 text-white hover:bg-rose-700 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Retry component
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
