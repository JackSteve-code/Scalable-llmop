import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  // ====================================================
  // 1. Site Metadata (Clean & Professional)
  // ====================================================
  title: 'JACK.SYSTEMS', 
  tagline: 'Software and ML Engineer specializing in AI/ML infrastructure',
  favicon: 'img/favicon.ico',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // ====================================================
  // 2. Deployment & Path Configuration
  // ====================================================
  url: 'https://JackSteve-code.github.io',
  baseUrl: '/Scalable-llmop/',
  organizationName: 'JackSteve-code',
  projectName: 'Scalable-llmop',
  trailingSlash: false,
  onBrokenLinks: 'warn',

  // ====================================================
  // 3. Presets & Plugins
  // ====================================================
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/', // Serves docs at the root
          editUrl: 'https://github.com/JackSteve-code/Scalable-llmop/tree/main/',
        },
        blog: false, 
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // ====================================================
  // 4. Theme Configuration (Toptal Style)
  // ====================================================
  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    
    navbar: {
      title: 'JACK.SYSTEMS', // Branding matches your portfolio
      logo: {
        alt: 'Engineering Logo',
        src: 'img/logo.svg', // Ensure you have a clean logo or use text-only
      },
      items: [
        {
          href: 'https://github.com/JackSteve-code/Scalable-llmop',
          label: 'GitHub Source',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'light', // Clean white/grey footer
      links: [], // Removed cluttered links
      copyright: `Copyright © ${new Date().getFullYear()} Jack Steve | AI/ML Infrastructure Expert`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
